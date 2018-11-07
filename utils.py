import numpy as np
import os
import csv
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

class TrainLogger():
    '''
        Log the trainig history.
        Usage: logger = TrainLogger("path_to_log.csv")
               logger.log(loss_train, loss_test)
        This class automatically append the csv file unless overwrite is set True.
    '''
    def __init__(self, dst_path, header=[], overwrite=False):
        self.path = dst_path
        if not os.path.exists(dst_path) or overwrite:
            with open(self.path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def log(self, row):
        with open(self.path, 'a') as f:
       	    writer = csv.writer(f)
            writer.writerow(row)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def infer(model, device, test_loader, dst_path):
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, names = data
            x = x.to(device)
            out = model(x)
            _, pred = torch.max(out.data, 1)
            with open(dst_path, 'a') as f:
                writer = csv.writer(f, delimiter="\t")
                for name, label in zip(names, pred):
                    writer.writerow([name,label.item()])



def test(model, device, test_loader):
    model.eval()
    correct = 0
    batch_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            batch_loss += model.loss(out, y).item()
            _, pred = torch.max(out.data, 1)
            correct += (y == pred).sum().item()

    loss = batch_loss/len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy


def train(model, device, train_loader, test_loader, optimizer, n_epochs,
                scheduler=None, done_epoch=0, prefix="", path_checkpoint="./checkpoint"):
    '''
        Method for training process.
        Args:
            done_epoch (int): if you resume the training, set the number of epochs you have done before.
            prefix (str): name the prefix for auto-saving files
            path_checkpoint (str): this method automatically saves parameters to 
                                   this location for every 10 epochs.
                                   
        For every epoch, parameters will be saved as "model.pth"
    '''
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    if done_epoch >= n_epochs:
        print("epochs exceeded{0}".format(n_epochs))
        return
    elif scheduler is not None:
        for i in range(done_epoch):
            scheduler.step()
    logger = TrainLogger("history-{}.csv".format(prefix))
    model.train()
    for epoch in range(done_epoch+1, n_epochs+1):
        print("epoch{}/{}".format(epoch, n_epochs))
        train_total_loss = 0
        for data in tqdm(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = model.loss(out, y)
            train_total_loss += loss.item()
            print(train_total_loss)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        train_loss = train_total_loss / len(train_loader) 
        test_loss, accuracy = test(model, device, test_loader)
        logger.log([train_loss, test_loss, accuracy])
        print("train:{}".format(train_loss))
        print("test:{}".format(test_loss))
        print("accuracy:{}\n---------".format(accuracy))
        save_model(model, "model.pth")
        save_model(optimizer, "opt.pth")
        if epoch % 10 == 0:
            model_name = "model-{0}-ep{1}.pth".format(prefix, epoch)
            opt_name = "opt-{0}-ep{1}.pth".format(prefix, epoch)

            save_model(model, os.path.join(path_checkpoint, model_name))
            save_model(optimizer, os.path.join(path_checkpoint, opt_name))

def plot_loss(train_loss, test_loss):
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"])
