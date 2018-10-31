
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from model import NIN 
from dataset import NINDataset
from utils import train, test

def main():
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

   # model = NIN()
    
    dataset_train = NINDataset("dict_class_neuron.pkl", "./train-file.lst", "./dataset")
    train_loader = DataLoader(dataset_train, batch_size=1000, shuffle=True)
    
    for i in train_loader:
        x, y = i
        r = x[:,0,:,:]
        g = x[:,1,:,:]
        b = x[:,2,:,:]
        print(r.mean().item(), g.mean().item(), b.mean().item())
        print(r.std().item(), g.std().item(), b.std().item())
        break

if __name__ == "__main__":
    main()
