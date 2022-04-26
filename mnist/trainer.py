import torch
from tqdm import tqdm
import torch.nn as nn
from model import Patch, Merge


class ModelLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1e-3):
        super(ModelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.standard_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, y):

        out = pred[0]  
        y1 = pred[1].argmax(-1)  
        y2 = pred[2] 
        W = pred[3]
        std_loss = self.standard_loss_fn(out, y1)  + self.alpha * self.standard_loss_fn(y2, y)
        loss = std_loss + self.beta * self.L21_norm(W)
        return loss

    def L21_norm(self, M):
        # sum (cols) over sum (rows)
        return torch.sum(torch.sqrt(torch.sum(M**2, dim=-1)))

def calculate_accuracy(pred, y):
  y_hat = pred.argmax(-1)  
  return (y == y_hat).sum(0) / len(y)

def train_epoch(train_x, train_y, model, optimizer, scheduler, loss_fn, device):
    n = train_x.size(0)
    model.to(device)
    model.train()
    losses = 0
    acc = 0
    cnt = 0 
    for i in tqdm(range(0, n, batch_size)):
        x = train_x[i: i + batch_size, ].to(device)
        y = train_y[i: i + batch_size].long().to(device)
        pred = model(x) 
        loss = loss_fn(pred, y)
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc += calculate_accuracy(pred[0], y)
        losses += loss.item()
        cnt += 1
return losses / cnt, acc / cnt


def val_epoch(test_x, test_y, model, loss_fn, device):
    n = test_x.size(0)
    model.to(device)
    model.eval()
    losses = 0
    acc = 0
    cnt = 0
    for i in tqdm(range(0, n, batch_size)):
        x = test_x[i: i + batch_size, ].to(device)
        y = test_y[i: i + batch_size].long().to(device)
        pred = model(x) 
        loss = loss_fn(pred, y)
        acc += calculate_accuracy(pred[0], y)
        losses += loss.item()
        cnt += 1
    return losses / cnt, acc / cnt

def infer(test_x, test_y, model, k)
    n = test_x.size(0)
    acc = 0
    for i in range(n):
        # Predict
        x = test_x[i: i + 1]
        pred = model(x)
        y = int(test_y[i].item())
        # Get weight vector
        weight = pred[-1][0, y, :]
        idx = torch.topk(weight, k).indices
        # Mask data
        e = Patch()(x).permute(1, 0, 2, 3)
        x_masked = torch.zeros_like(e)
        x_masked[:, idx, : , :] = e[:, idx, :, :]
        # Post-hoc evaluate
        img = Merge()(x_masked)
        y_hat = model.cls(img).argmax(-1).item()
        if y == y_hat:
            acc += 1
    return acc / n
    