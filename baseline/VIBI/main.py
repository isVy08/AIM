
import os, math
from utils import *
from tqdm import tqdm
import torch.nn as nn
from text_explainer import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_generator import Tokenizer, DataGenerator



def train_epoch(net, optimizer, scheduler, loader, class_criterion, info_criterion, X, Y, device):
    net.train()
    losses = 0
    accuracy = 0
    for idx in tqdm(loader): 
        # idx = next(iter(train_loader))
        x = X[idx, :].to(device)
        y = Y[idx].to(device)
        logit, log_p_i, Z_hat, logit_fixed = net(x)
        
        p_i_prior = prior(var_size = log_p_i.size()).to(device)

        class_loss = class_criterion(logit, y).div(math.log(2)) / len(idx)
        info_loss =  net.K * info_criterion(log_p_i, p_i_prior) / len(idx)
        total_loss = class_loss + net.beta * info_loss

        optimizer.zero_grad()
        total_loss.backward()        
        optimizer.step()
        if scheduler:
            scheduler.step()
        # print(list(net.parameters())[0].grad)
        acc = calculate_accuracy(logit, y)
        accuracy += acc
        losses += total_loss.item()

    return losses / len(loader), accuracy / len(loader)
   
 
def calculate_accuracy(pred, y):

    y_hat = pred.argmax(dim=1)
    acc = (y_hat==y).sum()            
    return acc / y.size(0)


def val_epoch(net, loader, class_criterion, info_criterion, X, Y, device):
    net.eval()
    losses = 0
    accuracy = 0
    for idx in tqdm(loader): 
        x = X[idx, :].to(device)
        y = Y[idx].to(device)
        logit, log_p_i, _, _  = net(x)
        p_i_prior = prior(var_size = log_p_i.size()).to(device)

        class_loss = class_criterion(logit, y).div(math.log(2))/ len(idx)
        info_loss =  net.K * info_criterion(log_p_i, p_i_prior) / len(idx) 
        total_loss = class_loss  + net.beta * info_loss

        acc = calculate_accuracy(logit, y)
        accuracy += acc
        
        losses += total_loss.item()
    return losses / len(loader), accuracy / len(loader)


def train(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    
    dg = DataGenerator(config)
    # dg.pad_idx = 1
    V = dg.tokenizer.get_vocab_size()
    dg.generate_data()

    # Load data
    train_indices = list(range(dg.train_x.size(0)))
    val_indices = list(range(dg.val_x.size(0)))

    train_loader = DataLoader(train_indices, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_indices, batch_size=config.batch_size, shuffle=False)


    dg.train_x = dg.train_x.flatten(1, 2)
    dg.val_x = dg.val_x.flatten(1, 2)

    
    val_x = Variable(dg.val_x)
    train_x = Variable(dg.train_x)
    val_y = Variable(dg.val_y.argmax(-1))
    train_y = Variable(dg.train_y.argmax(-1))



    # Load model, optimizer, loss function
    net = Explainer(V, config)

    if os.path.isfile(config.model_path):
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.5, 0.999))
        load_model(net, optimizer, config.model_path, device)
    else:
        net.weight_init()
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.5, 0.999))
        net.to(device)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    class_criterion = nn.CrossEntropyLoss(reduction='sum')
    info_criterion = nn.KLDivLoss(reduction='sum')
    
    print("Training begins")
    epochs = config.epochs
    prev_acc = 0.50

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(net, optimizer, None, train_loader, class_criterion, info_criterion, train_x, train_y, device)        
        
        # Evaluation
        val_loss, val_acc = val_epoch(net, val_loader, class_criterion, info_criterion, val_x, val_y, device)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f} // Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}"
        print(msg)

        if val_acc > prev_acc:
            print(f"Validation accuracy increases from {prev_acc:.3f} to {val_acc:.3f}. Saving model ...") 
            torch.save({'model_state_dict': net.state_dict() ,'optimizer_state_dict': optimizer.state_dict()}, config.model_path)
            prev_acc = val_acc

    
def validate(config):



    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    dg = DataGenerator(config)
    V = dg.tokenizer.get_vocab_size()
    
    test_x = dg._transform(dg.test_text)

    model = Explainer(V, config)
    load_model(model, None, config.model_path, device)

    score_file = open(config.score_path, 'w+')
    iter = range(test_x.size(0)) 
    acc = 0
    for i in tqdm(iter):
        text = dg.test_text[i]
        y = dg.test_label[i]

        x = test_x[i, :].to(device)
        x = x.flatten()
        x = x.unsqueeze(0)

        logit, score, _, _ = model(x)
        y_hat = logit.argmax().item()
        if y == y_hat:
            acc += 1
        
        out = (score.tolist(), y_hat)
        score_file.write(str(out)+'\n')


    print(f"Accuracy: {acc / len(iter)}")
    score_file.close()
    

if __name__ == "__main__":
    dataset = sys.argv[1]
    config_path = f'config/{dataset}.json'
    config = get_config(config_path)
    config.score_path = config.score_path + f'_k{config.K}.txt' 
    import sys
    if sys.argv[2] == 'train':
        train(config)
    else:
        dataset = config.data_path.split('/')[-2]
        validate(config)
        
