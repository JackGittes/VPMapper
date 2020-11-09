import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import tensorboardX as tb

from model.mobilenet import MobileNet_v1
from prune.init_param import init_model_theta
from prune.statistic import compute_sparsity
from prune.layer import PruneBatchNorm, penalty_loss
from prune.utils import replace_bn
from data.imagenet_seq.data import Loader as ImagenetLoader

os.environ['IMAGENET'] = '/home/ubuntu/user_space/Dataset/ILSVRC2012/'
os.environ['TENSORPACK_DATASET'] = '/home/ubuntu/user_space/Dataset/TENSORPACK/'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='weight/original/mobilenet_v1_1.0_224.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

""" Note this argument, if True, only apply validation. """
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')

parser.add_argument('--logdir', dest='logdir', default='logdir',
                    help='place to save checkpoint and also tensorboard logs')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--ngpus', default=-1, type=int,
                    help='-1 means all, 0 mean no gpu')

initial_param_path = 'weight/original/mobilenet_v1_1.0_224.pth'

trap_thresh = 0.05
target_ratio = 0.5
log_step = 0
prune_ratio = 0.43
start_point = 0.56242
p_weight = 0.005
alpha_0 = 5.0
feature_root = 'features'
summary_writer = tb.SummaryWriter('logdir')
save_folder = '/home/ubuntu/MyFiles/PruneMobileNetParams'


def main():
    global args, summary_writer
    args = parser.parse_args()

    # create tensorboard
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    global summary_writer
    summary_writer = tb.SummaryWriter(args.logdir)

    model = MobileNet_v1()
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_loader = ImagenetLoader('train', batch_size=512, num_workers=24, cuda=True)
    val_loader = ImagenetLoader('val', batch_size=512, num_workers=4, cuda=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    """ Get ready for pruning. """
    replace_bn(model)
    init_model_theta(model, feature_root, start_point, prune_ratio)
    for m_ in model.modules():
        if isinstance(m_, PruneBatchNorm):
            m_.thresh = trap_thresh

    masked = False
    epoch = 0
    tune_epoch = 0
    
    prune_lr = 0.002

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    optimizer = torch.optim.SGD(model.parameters(), prune_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_acc1_epoch = 0
    best_acc5_epoch = 0

    # change_optimizer = False
    learning_rate_reset = False
    while True:
        epoch += 1
        # train for one epoch
        masked = train(train_loader, model, criterion, optimizer, epoch, masked)

        if masked:
            if not learning_rate_reset:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                learning_rate_reset = True
            tune_epoch += 1
            adjust_learning_rate(optimizer, tune_epoch)
            if tune_epoch > args.epochs:
                break

        # evaluate on validation set
        acc1, acc5, loss = validate(val_loader, model, criterion)

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc1_epoch = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_acc5_epoch = epoch

        summary_writer.add_scalar('val/prec1', acc1, epoch)
        summary_writer.add_scalar('val/prec5', acc5, epoch)
        summary_writer.add_scalar('val/loss', loss, epoch)
        # remember best prec@1 and save checkpoint
        torch.save({'state_dict': model.state_dict(),
                    'acc1': float(acc1),
                    'acc5': float(acc5),
                    'best_acc1': float(best_acc1),
                    'best_acc1_epoch': int(best_acc1_epoch),
                    'best_acc5': float(best_acc5),
                    'best_acc5_epoch': int(best_acc5_epoch)
                    },
                   os.path.join(save_folder, '{:>03d}.pth'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch, masked):
    global log_step, p_weight, summary_writer, target_ratio

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (im, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target + 1

        # compute output
        output = model(im)
        loss_c = criterion(output, target)
        loss_p = penalty_loss(model, alpha_0) * p_weight

        loss = loss_c + loss_p

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sp = compute_sparsity(model, trap_thresh, masked, False, True)
        if not masked:
            if sp >= target_ratio:
                masked = True
                p_weight = 0.0
                for m_ in model.modules():
                    if isinstance(m_, PruneBatchNorm):
                        m_.mask_gradient()
                compute_sparsity(model, trap_thresh, masked, True, True)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        summary_writer.add_scalar('train/prec1', scalar_value=acc1[0], global_step=log_step)
        summary_writer.add_scalar('train/prec5', scalar_value=acc5[0], global_step=log_step)
        summary_writer.add_scalar('train/cls_loss', scalar_value=float(loss_c), global_step=log_step)
        summary_writer.add_scalar('train/sparsity', scalar_value=sp, global_step=log_step)
        summary_writer.add_scalar('train/p_loss', scalar_value=float(loss_p), global_step=log_step)
        losses.update(loss.item(), im.size(0))
        top1.update(acc1[0], im.size(0))
        top5.update(acc5[0], im.size(0))

        log_step += 1
        # print(log_step)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return masked


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (im, target) in enumerate(val_loader):
            # target = target.cuda(non_blocking=True)
            target = target + 1
            # im = im.cuda()
            # compute output
            output = model(im)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), im.size(0))
            top1.update(prec1[0], im.size(0))
            top5.update(prec5[0], im.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.logdir, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.logdir, filename), os.path.join(args.logdir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
