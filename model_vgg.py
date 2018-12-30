import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from tqdm import tqdm

import torchvision.models as models


'''
VGG16
'''

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.vgg_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.vgg_conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.vgg_conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg_pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc1 = nn.Linear(8*8*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 11)
        
    
    def forward(self, x):
        x = F.relu(self.vgg_conv1(x))
        x = F.relu(self.vgg_conv2(x))
        x = self.vgg_pool1(x)
        x = F.relu(self.vgg_conv3(x))
        x = F.relu(self.vgg_conv4(x))
        x = self.vgg_pool2(x)
        x = F.relu(self.vgg_conv5(x))
        x = F.relu(self.vgg_conv6(x))
        x = F.relu(self.vgg_conv7(x))
        x = self.vgg_pool3(x)
        x = F.relu(self.vgg_conv8(x))
        x = F.relu(self.vgg_conv9(x))
        x = F.relu(self.vgg_conv10(x))
        x = self.vgg_pool4(x)
        x = F.relu(self.vgg_conv11(x))
        x = F.relu(self.vgg_conv12(x))
        x = F.relu(self.vgg_conv13(x))
        x = self.vgg_pool5(x)
        x = x.view(-1, 8*8*512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def loss(self, y, t):
        crossentropy = nn.CrossEntropyLoss()
        total_loss = crossentropy(y, t) 
        return total_loss
