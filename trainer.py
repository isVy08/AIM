import random
import torch
from tqdm import tqdm
from utils_eval import mask_data


def train_epoch(model, optimizer, scheduler, loader, loss_fn, X, Y, device):
    model.train()
    losses = 0
    accuracy = 0
    for idx in tqdm(loader): 
        x = X[idx, :].to(device)
        y = Y[idx].to(device)
        
        pred = model(x)
        if y.size(1) > 1:
            y = y.argmax(-1)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
            
        loss.backward()
        
        optimizer.step()
        if scheduler:
            scheduler.step()

        acc = calculate_accuracy(pred, y)
        accuracy += acc
        losses += loss.item()
    return losses / len(loader), accuracy / len(loader)
   


def calculate_accuracy(pred, y):
    if len(pred.shape) > 2:
        y_hat = pred[:, :, 2].argmax(-1)
    else:
        y_hat = pred.argmax(-1)
    return (y == y_hat).sum(0) / len(y)
    

def val_epoch(model, loader, loss_fn, X, Y, device):
    model.eval()
    losses = 0
    accuracy = 0
    for idx in tqdm(loader):
        x = X[idx, :].to(device)
        y = Y[idx].to(device)
        pred = model(x)

        if y.size(1) > 1:
            y = y.argmax(-1)

        loss = loss_fn(pred, y) 
        accuracy += calculate_accuracy(pred, y)
        losses += loss.item()

    return losses / len(loader), accuracy / len(loader)


