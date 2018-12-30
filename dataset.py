import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
import cv2

class MyDataset(Dataset):
    '''
        Args:
        csv_path (string): Specify the location of csv file including pairs of relative paths.
        root_dir (string): Specify the root directory used in csv_path.
    '''
    def __init__(self, pickle_path, csv_path, root_dir, transform=None, infer=False, canny=False):
        self.file_list = pd.read_csv(csv_path, delimiter=",", header=None)
        self.root_dir = root_dir
        self.transform = transform

        self.mean = [0.154, 0.208, 0.070]
        self.std = [0.068, 0.065, 0.051]
        self.infer = infer
        self.canny = canny

        with open("dict_class_neuron.pkl" ,"rb") as f:
            self.dict = pickle.load(f)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        x_img_name = os.path.join(self.root_dir, self.file_list.iloc[idx,0])
        x_img = Image.open(x_img_name)
        x_img = np.asarray(x_img, dtype=np.float32)
        x_img = x_img.transpose([2,0,1])
        x_img /= 255.0
        x_img[0,:,:] = (x_img[0,:,:] - self.mean[0]) / self.std[0]
        x_img[1,:,:] = (x_img[1,:,:] - self.mean[1]) / self.std[1]
        x_img[2,:,:] = (x_img[2,:,:] - self.mean[2]) / self.std[2]
        

        y_label = self.file_list.iloc[idx,1]
        y_label = self.dict[y_label]
        if self.infer:
            x_img = torch.tensor(x_img, dtype=torch.float)
            img_name = os.path.basename(self.file_list.iloc[idx,0])
            return (x_img, img_name)
        elif self.canny:
            img = x_img.transpose([1,2,0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[np.newaxis, :, :]
            x_img = np.vstack((x_img, img))
            x_img = torch.tensor(x_img, dtype=torch.float)
            return (x_img, y_label)
        else:
            x_img = torch.tensor(x_img, dtype=torch.float)
            return (x_img, y_label)

