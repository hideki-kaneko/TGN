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

def main(load_model=False):
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    if load_model:
        model = NIN()
        model.load_state_dict(torch.load("nin.model"))
    else:
        model = NIN()
    
    dataset_test = NINDataset("dict_class_neuron.pkl", "./test-file.lst", "./dataset")
    test_loader = DataLoader(dataset_test, batch_size=16, shuffle=True)
    
    device = torch.device("cuda:0")
    model = model.to(device)
    
    loss, acc = test(model=model, device=device, test_loader=test_loader)
    print(loss, acc)

if __name__ == "__main__":
    main(load_model=True)
