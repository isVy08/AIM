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
        if y.size(1) == 2:
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
   

def calculate_ph_accuracy(x, y, pred, model, device, s):   
    acc = 0 
    L = model.L
    N = x.size(0)

    idx = random.sample(range(N), s)
    for i in idx:
        y0 = y[i,]
        score = pred[i, y0, 3:]
        masked_x = mask_data(score, x[i, :], 10, 'neg', device, L)
        masked_x = masked_x.unsqueeze(0)
        masked_pred = model.cls(masked_x)
        masked_y_hat = masked_pred.argmax(-1).item()
        if masked_y_hat == [i].item():
            acc += 1
    return acc

def calculate_accuracy(pred, y):
    if len(pred.shape) > 2:
        y_hat = pred[:, :, 2].argmax(-1)
    else:
        y_hat = pred.argmax(-1)
    return (y == y_hat).sum(0) / len(y)
    

def val_epoch(model, loader, loss_fn, X, Y, device):
    model.eval()
    losses = 0
    accuracy, ph_accuracy, cnt = 0, 0, 0
    for idx in tqdm(loader): 
        s = len(idx) // 3
        if s == 0:
            s = len(idx)
        cnt += s
        x = X[idx, :].to(device)
        y = Y[idx].to(device)
        pred = model(x)
        
        if y.size(1) == 2:
            y = y.argmax(-1)

        loss = loss_fn(pred, y) 
        if len(pred.shape) > 2:
            # Evaluate faithfulness for top 10
            ph_acc = calculate_ph_accuracy(x, y, pred, model, device, s)
            ph_accuracy += ph_acc
        else:
            ph_accuracy += 0
        accuracy += calculate_accuracy(pred, y)
        losses += loss.item()

    return losses / len(loader), accuracy / len(loader), ph_accuracy / cnt



