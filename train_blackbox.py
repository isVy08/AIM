import os
import torch
from tqdm import tqdm
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_generator import DataGenerator, Tokenizer
from trainer import train_epoch, val_epoch, calculate_accuracy



def train(config):

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # load data
    dg = DataGenerator(config)
    dg.generate_data()

    train_indices = list(range(dg.train_x.size(0)))
    val_indices = list(range(dg.val_x.size(0)))

    train_loader = DataLoader(train_indices, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_indices, batch_size=config.batch_size, shuffle=False)

    V = dg.tokenizer.get_vocab_size()
    
    # load model, optimizer, loss function
    if config.model_name == 'WordGRU':
        from blackbox import WordGRU
        model = WordGRU(V)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        
    elif config.model_name == 'WordCNN':
        from blackbox import WordCNN
        model = WordCNN(V) 
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
    
    elif config.model_name == 'WordTF':
        from blackbox import WordTransformer
        model = WordTransformer(V, L = config.max_length)
        model.to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)  
    
    if os.path.isfile(config.model_path):
        load_model(model, optimizer, config.model_path, device)
    
 
    model.to(device)
    model.train()
    

    dg.train_x = dg.train_x.to(device)
    dg.val_x = dg.val_x.to(device)
    dg.train_y = dg.train_y.long()
    dg.val_y = dg.val_y.long()
    
    # training
    print("Training begins")
    loss_fn =  nn.CrossEntropyLoss()
    
    epochs = config.epochs
    prev_acc = 0.50
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, optimizer, scheduler, train_loader, loss_fn, dg.train_x, dg.train_y, device)        
        
        # Evaluation
        val_loss, val_acc = val_epoch(model, val_loader, loss_fn, dg.val_x, dg.val_y, device)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f} // Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}"
        print(msg)

        if val_acc > prev_acc:
            print(f"Validation accuracy increases from {prev_acc:.3f} to {val_acc:.3f}. Saving model ...") 
            torch.save({'model_state_dict': model.state_dict() ,'optimizer_state_dict': optimizer.state_dict(),}, config.model_path)
    
            prev_acc = val_acc

def validate(model, loader, X, Y, device, file):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for idx in tqdm(loader): 
            x = X[idx, :].to(device)
            y = Y[idx].to(device)
            pred = model(x)
            acc = calculate_accuracy(pred, y)
            accuracy += acc
            for i in pred.tolist():
                file.write(str(i) + '\n')
    return accuracy / len(loader)

    


def predict(config):

     
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
     
    # load data
    dg = DataGenerator(config)
    dg.generate_data()

    train_indices = list(range(dg.train_x.size(0)))
    val_indices = list(range(dg.val_x.size(0)))
    test_indices = list(range(dg.test_x.size(0)))

    train_loader = DataLoader(train_indices, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_indices, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_indices, batch_size=config.batch_size, shuffle=False)
    
    V = dg.tokenizer.get_vocab_size()
    
    # load model, optimizer, loss function
    if config.model_name == 'WordGRU':
        from blackbox import WordGRU
        model = WordGRU(V, C = config.C)
    elif config.model_name == 'WordCNN':
        from blackbox import WordCNN
        model = WordCNN(V, C = config.C) 
    elif config.model_name == 'WordTF':
        from blackbox import WordTransformer
        model = WordTransformer(V, L = config.max_length, C = config.C)
    
    if os.path.isfile(config.model_path):
        load_model(model, None, config.model_path, device)
    
    dg.train_y = dg.train_y.long()
    dg.val_y = dg.val_y.long()
    dg.test_y = dg.test_y.long()
    
    file = open(config.output_path[0], 'w+')
    train_acc = validate(model, train_loader, dg.train_x, dg.train_y, device, file)
    val_acc = validate(model, val_loader, dg.val_x, dg.val_y, device, file)
    test_acc = validate(model, test_loader, dg.test_x, dg.test_y, device, file)
    
    file.close()
    print(f'Training Accuracy: {train_acc:.4f} // Validation Accuracy: {val_acc:.4f} // Test Accuracy: {test_acc:.4f}')

    predictions = load(config.output_path[0])
    predictions = [eval(pred) for pred in predictions]
    assert len(predictions) == len(dg.train_text) + len(dg.val_text) + len(dg.test_text)
    N = len(dg.train_text)
    data = {
        "train": (dg.train_text, predictions[:N]), 
        "val": (dg.val_text + dg.test_text, predictions[N:] )
        }
    write_pickle(data, config.output_path[1])



if __name__ == "__main__":
    import sys
    config_file = sys.argv[1]
    config = get_config(config_file)
    if sys.argv[2] == 'train':
        train(config)
    elif sys.argv[2] == 'val':
        predict(config)
    else: 
        raise ValueError("Unknown operation")