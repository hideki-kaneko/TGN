import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import math

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

# https://github.com/yueruchen/sppnet-pytorch
def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
 '''
 previous_conv: a tensor vector of previous convolution layer
 num_sample: an int number of image in the batch
 previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
 out_pool_size: a int vector of expected output size of max pooling layer

 returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
 '''    
 # print(previous_conv.size())
 for i in range(len(out_pool_size)):
     # print(previous_conv_size)
     h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
     w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
     h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
     w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
     maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
     x = maxpool(previous_conv)
     if(i == 0):
         spp = x.view(num_sample,-1)
         # print("spp size:",spp.size())
     else:
         # print("size:",spp.size())
         spp = torch.cat((spp,x.view(num_sample,-1)), 1)
 return spp

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
        self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1)
        self.drop3 = nn.Dropout2d(0.5)
        
        self.pool3 = nn.AvgPool2d(64, stride=1) # global average pooling
        
        # side
        self.s_conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=1, bias=False)

        # merge
        self.dense1 = nn.Linear(256*256*2, 2048)
        self.dense2 = nn.Linear(2048, 2048)
        self.dense3 = nn.Linear(2048, 29)

    def forward(self, x):
        canny, img = torch.split(x, [1,3], dim=1)

        # side = self.s_conv1(canny)
        # side = spatial_pyramid_pool(self, side, side.shape[0], side.shape[2:], [2,4,8])
        side = canny.view(-1, 256*256*1)

        x = F.leaky_relu(self.drop1(self.bn1(self.conv1(img))), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.drop2(self.bn2(self.conv2(x))), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.drop3(self.bn3(self.conv3(x))), negative_slope=0.1, inplace=True)
        # x = spatial_pyramid_pool(self, x, x.shape[0], x.shape[2:], [2,4,8]) #1344
        x = x.view(-1,256*256*1)

        merge = torch.cat((x, side), dim=1)
        merge = F.leaky_relu(self.dense1(merge), negative_slope=0.1, inplace=True)
        merge = F.leaky_relu(self.dense2(merge), negative_slope=0.1, inplace=True)
        merge = F.leaky_relu(self.dense3(merge), negative_slope=0.1, inplace=True)

        y = F.softmax(merge, dim=1)
        return y

    def loss(self, y, t):
        crossentropy = nn.CrossEntropyLoss()
        total_loss = crossentropy(y, t) 
        return total_loss

