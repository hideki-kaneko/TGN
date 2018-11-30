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

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(0.5)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(0.5)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(32, 11, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(11)
        self.drop3 = nn.Dropout2d(0.5)
        self.pool3 = nn.AvgPool2d(64, stride=1) # global average pooling
        
    def forward(self, x):
        x = F.leaky_relu(self.pool1(self.drop1(self.bn1(self.conv1(x)))), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.pool2(self.drop2(self.bn2(self.conv2(x)))), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.pool3(self.drop3(self.bn3(self.conv3(x)))), negative_slope=0.1, inplace=True)
        x = x.view(-1, 11)
        x = F.softmax(x, dim=1)
        return x

    def loss(self, y, t):
        crossentropy = nn.CrossEntropyLoss()
        total_loss = crossentropy(y, t) 
        return total_loss

