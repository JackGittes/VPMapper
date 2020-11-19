import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self, gpu_id, train=True, lite=True):
        super(SiameseAlexNet, self).__init__()
        if not lite:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 11, 2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(96, 256, 5, 1, groups=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(256, 384, 3, 1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, 3, 1, groups=2),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, 3, 1, groups=2)
            )
        else:
            self.features = nn.Sequential(
                # First Layer of AlexNet #
                nn.Conv2d(3, 96, kernel_size=11, stride=2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                # Second Layer #
                nn.MaxPool2d(3, 2),

                # Third Layer #
                nn.Conv2d(96, 96, kernel_size=5, stride=1, groups=96),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                # Fourth Layer #
                nn.MaxPool2d(kernel_size=3, stride=2),

                # Fifth Layer #
                nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=256),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, kernel_size=1, stride=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),

                # Sixth Layer #
                nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=1, stride=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),

                # Output Layer #
                nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=384),
                nn.BatchNorm2d(384),
                nn.ReLU(384),

                nn.Conv2d(384, 256, kernel_size=1, stride=1)
            )
        self.corr_bias = nn.Parameter(torch.zeros(1))
        if train:
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz))
            with torch.cuda.device(gpu_id):
                self.train_gt = torch.from_numpy(gt).cuda()
                self.train_weight = torch.from_numpy(weight).cuda()
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
            with torch.cuda.device(gpu_id):
                self.valid_gt = torch.from_numpy(gt).cuda()
                self.valid_weight = torch.from_numpy(weight).cuda()
        self.exemplar = None
        self.gpu_id = gpu_id

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            exemplar = self.features(exemplar)
            instance = self.features(instance)
            n, ch, h, w = instance.shape
            instance = instance.view(1, -1, h, w)
            score = F.conv2d(instance, exemplar, groups=n) * config.response_scale + self.corr_bias
            return score.transpose(0, 1)
        elif exemplar is not None and instance is None:
            # inference used
            self.exemplar = self.features(exemplar)
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)
        else:
            # inference used we don't need to scale the response or add bias
            instance = self.features(instance)
            n, _, h, w = instance.shape
            instance = instance.view(1, -1, h, w)
            score = F.conv2d(instance, self.exemplar, groups=n)
            return score.transpose(0, 1)

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                                                      self.train_weight, reduction='sum') / \
                   config.train_batch_size  # normalize the batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                                                      self.valid_weight, reduction='sum') / \
                   config.train_batch_size  # normalize the batch_size

    @staticmethod
    def _create_gt_mask(shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)

