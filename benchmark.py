import argparse
import os
import pstats
from cProfile import Profile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from torchvision import models


'''
    A script for testing computation time.

'''

def count_params(model):
    total = 0
    for i in model.parameters():
        total += np.prod(i.shape)
    return total

def main(args):
    SEED = 12345

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    
    model = models.vgg16(pretrained=True)
    model.classifier[0] = nn.Linear(8*8*512, 4096)
    model.classifier[6] = nn.Linear(4096, 27)
    model.load_state_dict(torch.load("model.pth"))
    print("total params:{}".format(count_params(model)))    

    dataset_test = MyDataset(args.pickle_path, args.test_list_path, args.test_dir)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cpu" if args.no_cuda else "cuda:0")
    model = model.to(device)
    
    model.eval()

    if not args.no_prof:
        pr = Profile()
        pr.runcall(bench, model, test_loader, device)
        pr.dump_stats("profile.txt")

        ps = pstats.Stats("profile.txt")
        ps.sort_stats('time').print_stats(15)


def bench(model, test_loader, device):
    print("start benchmark...")
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            _, pred = torch.max(out.data, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list_path", required=True, help="location of test files list")
    parser.add_argument("--test_dir", required=True, help="location of test files directory")
    parser.add_argument("--pickle_path", required=True, help="location of pickle file")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--model_path", required=True, help="location of saved weight")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU")
    parser.add_argument("--no-prof", action="store_true", help="disable Profiling")

    args = parser.parse_args()

    main(args)
