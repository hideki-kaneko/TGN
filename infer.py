import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from torchvision import models


def main(args):
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    model = models.vgg16(pretrained=True)
    model.classifier[0] = nn.Linear(8*8*512, 4096)
    model.classifier[6] = nn.Linear(4096, 27)
    model.load_state_dict(torch.load("model.pth"))
    
    dataset_test = MyDataset(args.pickle_path, args.test_list_path, args.test_dir)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cpu" if args.no_cuda else "cuda:0")
    model = model.to(device)
    
    model.eval()
    correct = 0
    with open(args.result_path, 'w') as f:
        writer = csv.writer(f)
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                _, pred = torch.max(out.data, 1)
                result = torch.stack((y, pred), dim=1).cpu().numpy()
                writer.writerows(result)
                correct += (y == pred).sum().item()

    # loss = batch_loss/len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list_path", required=True, help="location of test files list")
    parser.add_argument("--test_dir", required=True, help="location of test files directory")
    parser.add_argument("--pickle_path", required=True, help="location of pickle file")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--result_path", default="./result.csv", help="output location")

    parser.add_argument("--model_path", required=True, help="location of saved weight")
    parser.add_argument("--optimizer_path", default="opt.pth", help="parameters updated every epoch")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU")
    parser.add_argument("--num_workers", type=int, default=1, help="number of threads for dataloader")

    args = parser.parse_args()

    main(args)
