from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import argparse

from dataset import MyDataset
from utils import train, test, TrainLogger

'''
    A script for training.

'''

SEED = 1

def main(args):
    canny = False
    writer = None
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    done_epoch = 0

    if args.model_type == "nin":
        from model_nin import MyModel
        model = MyModel()
    elif args.model_type == "tgn":
        from model_tgn import MyModel
        model = MyModel()
        canny = True
    elif args.model_type == "vgg":
        def loss(y, t):
            crossentropy = nn.CrossEntropyLoss()
            total_loss = crossentropy(y, t) 
            return total_loss
        model = models.vgg16(pretrained=True)
        model.classifier[0] = nn.Linear(8*8*512, 4096)
        model.classifier[6] = nn.Linear(4096, 27)
        model.loss = loss 

    if args.resume and os.path.exists(args.model_path):
        print("Resume training...")
        # model = Generator(Encorder(), Decoder())
        model.load_state_dict(torch.load(args.model_path))
        with open("{0}-history.csv".format(args.expname), 'r') as f:
            for i, l in enumerate(f):
                pass
            done_epoch = i
    else:
        print("Start training from scratch...")
        # model = Generator(Encorder(), Decoder())
        HEADER = ["loss_train", "loss_test", "accuracy"]
        logger = TrainLogger("{0}-history.csv".format(args.expname), header=HEADER, overwrite=True)
        del(logger)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED) 

    dataset_train = MyDataset(pickle_path=args.pickle_path, csv_path=args.train_list_path, root_dir=args.train_dir, canny=canny)
    dataset_test = MyDataset(pickle_path=args.pickle_path, csv_path=args.test_list_path, root_dir=args.test_dir, canny=canny)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device("cpu" if args.no_cuda else "cuda:0")
    model = model.to(device)
    optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if args.resume:
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    train(model=model, device=device, train_loader=train_loader, test_loader=test_loader,
                                  optimizer=optimizer, n_epochs=args.n_epochs, prefix=args.expname, done_epoch=done_epoch, path_checkpoint="checkpoint-{}".format(args.expname), writer=writer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", required=True, help="prefix for auto-saving params and loss history")
    parser.add_argument("--train_list_path", required=True, help="location of trainig files list")
    parser.add_argument("--train_dir", required=True, help="location of trainig files directory")
    parser.add_argument("--test_list_path", required=True, help="location of test files list")
    parser.add_argument("--test_dir", required=True, help="location of test files directory")
    parser.add_argument("--pickle_path", required=True, help="location of pickle file")

    parser.add_argument("--model_type", choices=['nin', 'tgn', 'vgg'], default="nin", help="model type")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum factor")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")

    parser.add_argument("--model_path", default="hed.model", help="parameters updated every epoch")
    parser.add_argument("--optimizer_path", default="opt.pth", help="parameters updated every epoch")
    parser.add_argument("--resume", action="store_true", help="resume the training")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU")
    parser.add_argument("--num_workers", type=int, default=1, help="number of threads for dataloader")
    parser.add_argument("--tensorboard", action="store_true")

    args = parser.parse_args()

    main(args)
