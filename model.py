import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLPConv(nn.Module):
    def __init__(self,ch_in, ch1, ch2, ch_out):
        super(MLPConv, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch1, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=1, padding=0, stride=1)
        self.conv3 = nn.Conv2d(ch2, ch_out, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class NIN(nn.Module):
    def __init__(self):
        super(NIN, self).__init__()
        self.mlpc1 = MLPConv(3, 192, 192, 192)
        self.drop1 = nn.Dropout2d(0.5)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.mlpc2 = MLPConv(192, 192, 192, 192)
        self.drop2 = nn.Dropout2d(0.5)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.mlpc3 = MLPConv(192, 192, 192, 11)
        self.pool3 = nn.AvgPool2d(64, stride=1) # global average pooling
        
    def forward(self, x):
        x = self.mlpc1(x)
        x = self.pool1(self.drop1(x))
        x = self.mlpc2(x)
        x = self.pool2(self.drop2(x))
        x = self.mlpc3(x)
        x = self.pool3(x)
        x = x.view(-1, 11)
        x = F.softmax(x, dim=1)
        return x

class TGN(nn.Module):
    def __init(self):
        super.conv1 = 
