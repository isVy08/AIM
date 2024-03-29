import math
import torch, os
from utils import *
from tqdm import tqdm
from trainer import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from explainer import Model
from data_generator import Tokenizer, DataGenerator

torch.manual_seed(8)

class ModelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(ModelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.standard_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, y):

        out = pred[:, :, 0] # Explainer 
        y1 = pred[:, :, 1].argmax(-1) # Mimic black-box
        y2 = pred[:, :, 2] # Maximize Mutual Information
        W = pred[:, :, 3:]
        std_loss = self.standard_loss_fn(out, y1)  + self.alpha * self.standard_loss_fn(y2, y)
        loss = std_loss + self.beta * self.L21_norm(W)
        return loss

    def L21_norm(self, M):
        # sum (cols) over sum (rows)
        return torch.sum(torch.sqrt(torch.sum(M**2, dim=-1)))

def train(config):


    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    cls_config = get_config(config.classifier_config)


    # Obtain black box predictions as ground-truth labels (!!!)
    cls_config.data_path = config.data_path
    
    D = config.D
    C = config.C
    HS, HE, HU = config.HS, config.HE, config.HU
    p = config.dropout
    tau = config.tau
    in_kernel = config.in_kernel
    alpha, beta = config.alpha, config.beta
    L = cls_config.max_length
    

    dg = DataGenerator(cls_config)
    dg.data_fraction = config.data_fraction
    print(dg.data_path)
    dg.generate_data()

    V = dg.tokenizer.get_vocab_size()
    train_indices = list(range(dg.train_x.size(0)))

    # load classifier
    
    if cls_config.model_name == 'WordGRU':
        from blackbox import WordGRU
        cls = WordGRU(V)
    elif cls_config.model_name == 'WordCNN':
        from blackbox import WordCNN
        cls = WordCNN(V) 
        train_indices = list(range(dg.train_x.size(0) // 2))
    elif cls_config.model_name == 'WordTF':
        from blackbox import WordTransformer
        cls = WordTransformer(V, L = cls_config.max_length, C = cls_config.C)
    
    # Use temparture softmax on black-box if needed. We currently use temp = 1
    cls.temp = config.temp 
    
    val_indices = list(range(dg.val_x.size(0)))
    train_loader = DataLoader(train_indices, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_indices, batch_size=config.batch_size, shuffle=False)


    load_model(cls, None, cls_config.model_path, device)
    cls.eval()

    
    # load model, optimizer, loss function
    model = Model(V, D, L, C, HS, HE, HU, in_kernel, device, p, tau, cls)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)  
    
    if os.path.isfile(config.model_path):
        load_model(model, optimizer, config.model_path, device)
    
    
    model.train()
    model.to(device)

    dg.train_x = dg.train_x.to(device)
    dg.val_x = dg.val_x.to(device)
        
    # training
    print("Training begins")
    epochs = config.epochs
    prev_acc = 0.20 ## edit this

    loss_fn = ModelLoss(alpha, beta)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, optimizer, scheduler, train_loader, loss_fn, dg.train_x, dg.train_y, device)        
        
        # Evaluation
        val_loss, val_acc  = val_epoch(model, val_loader, loss_fn, dg.val_x, dg.val_y, device)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f} // Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}"
        print(msg)

        if math.isnan(train_loss) or math.isnan(val_loss):
            break
            
        if val_acc >= prev_acc: 
            print(f"Validation accuracy increases from {prev_acc:.3f} to {val_acc:.3f}. Saving model ...") 
            torch.save({'model_state_dict': model.state_dict() ,'optimizer_state_dict': optimizer.state_dict(),}, config.model_path)
    
            prev_acc = val_acc
        
    
if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]
    config_path = f'config/{dataset}.json'
    config = get_config(config_path)
    no = sys.argv[2]
    config.model_path = config.model_path + f'_{no}.pt'
    
    train(config)