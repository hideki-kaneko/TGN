import numpy as np
import os
import csv
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sendline import LineSender

class TrainLogger():
    def __init__(self, dst_path, header, overwrite=False):
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
    correct = 0
    total_loss = 0
    crossentropy = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_loss += crossentropy(out, y).item() # detach the history
            _, pred = torch.max(out.data, 1)
            correct += (y == pred).sum().item()

    loss = total_loss/len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy

def train(model, device, train_loader, test_loader, optimizer, n_epochs):
    crossentropy = nn.CrossEntropyLoss()
    best_score = 0
    logger = TrainLogger("history.csv", ["loss_train", "loss_test", "accuracy"])
                
    sender = LineSender("history.csv")

    for epoch in range(n_epochs):
        train_total_loss = 0
        for data in tqdm(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = crossentropy(out, y)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_total_loss / len(train_loader.dataset) 
        test_loss, accuracy =  test(model, device, test_loader)
        logger.log([train_loss, test_loss, accuracy])
        save_model(model, "nin.model")
        if epoch %1 == 0:
            sender.post_loss()
        if accuracy > best_score:
            save_model(model, "best.model")
            best_score = accuracy 

    print("best:", best_acc)

    
def plot_loss(train_loss, test_loss):
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"])
