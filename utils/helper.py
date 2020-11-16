import time
import torch
import numpy as np


def file_id():
    """
    Generate a log folder for saving training checkpoint,
    if log_root is provided, the folder is created in log_root with
    log_time_random number as its name.

    :return: None
    """
    time_stamp = time.strftime("%Y-%m-%dd-%Hh-%Mm-%Ss", time.localtime())

    random_id = np.random.randint(0, 99999)
    return time_stamp+"_"+"{:<05d}".format(random_id)


def check_tuple(_in)->(int, int):
    assert isinstance(_in, int) or isinstance(_in, tuple), \
        "Input must be a tuple or an integer."
    if isinstance(_in, tuple):
        res = _in
    else:
        res = (_in, _in)
    return res


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


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


