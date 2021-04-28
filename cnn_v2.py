import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np



class ConvBlock(nn.Module):
    def __init__(self, in_maps, maps, step=1):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_maps, maps, kernel_size=3,
                      padding=1,stride=step,bias=False),
            nn.BatchNorm2d(maps),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(maps, maps, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(maps),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class SkipBlock(nn.Module):

    def __init__(self, in_maps, maps, step=1):
        super(SkipBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_maps, maps, kernel_size=3,
                      padding=1,stride=step,bias=False),
            nn.BatchNorm2d(maps),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(maps, maps, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(maps),
            nn.LeakyReLU(),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_maps, maps,
                      kernel_size=1, stride=step, bias=False),
            nn.BatchNorm2d(maps)
        )

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += self.skip(identity)
        return out


class I_encoder(nn.Module):
    def __init__(self, net_size=18):
        super(I_encoder, self).__init__()
        self.blocks = []
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            *self._add_block(64, 64, 1)
        )
        self.layer3 = nn.Sequential(
            *self._add_skip_block(64, 100, 1)
        )
        self.layer4 = nn.Sequential(
            *self._add_skip_block(100, 150, 1)
        )
        self.layer5 = nn.Sequential(
            *self._add_skip_block(150, 150, 1)
        )
        self.dropout = nn.Dropout(p=0.25)


    def _add_block(self, in_maps, maps, step=1):
        # cnn_block = ConvBlock(in_maps, maps, step=1)
        layers = []
        cnn_block = ConvBlock(in_maps, maps, step=1)
        layers.append(cnn_block)
        return layers

    def _add_skip_block(self, in_maps, maps, step=2):
        skip_block = SkipBlock(in_maps, maps, step=2)
        layers = [skip_block]
        return layers

    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = nn.MaxPool2d(2)(out)
        return out

