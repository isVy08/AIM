import torch
from utils import *
from tqdm import tqdm
from utils_eval import *
from explainer import Model, Selector, Explainer
from data_generator import Tokenizer, DataGenerator



def infer(config, score_file): 
    """
    config      : model configurations
    k           : k max for early stopping
    output_file : write qualitative samples for top k features
    score_file  : write weight vectors for conventional inference 
    """

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    cls_config = get_config(config.classifier_config)

    # Remember to evaluate on black-box predictions
    cls_config.data_path = config.data_path 
    
    D = config.D
    C = config.C
    HS, HE, HU = config.HS, config.HE, config.HU
    p = config.dropout
    tau = config.tau
    in_kernel = config.in_kernel
    L = cls_config.max_length
    
    dg = DataGenerator(cls_config)
    
    print(dg.data_path)

    V = dg.tokenizer.get_vocab_size()
    test_x = dg._transform(dg.test_text)


    # load Classifier
    if cls_config.model_name == 'WordGRU':
        from blackbox import WordGRU
        cls = WordGRU(V)

    elif cls_config.model_name == 'WordCNN':
        from blackbox import WordCNN
        cls = WordCNN(V) 

    elif cls_config.model_name == 'WordTF':
        from blackbox import WordTransformer
        cls = WordTransformer(V, L = cls_config.max_length, C = cls_config.C)

    
    load_model(cls, None, cls_config.model_path, device)
    cls.eval()
    cls.temp = config.temp


    model = Model(V, D, L, C, HS, HE, HU, in_kernel, device, p, tau, cls)
    load_model(model, None, config.model_path, device)
    model.eval()

    iter = range(test_x.size(0)) 

    A = 0
    
    for i in tqdm(iter):
        y = np.argmax(dg.test_label[i], -1)

        x = test_x[i, :].to(device)
        x = x.unsqueeze(0)

        pred = model(x)
        y_hat = pred[:, :, 0].argmax().item() # explainer's predictions
        weight = pred[0, y, 3:] # spatial weight W

        # Write weight vectors for conventional inference
        if score_file is not None:
            out = (weight.tolist(), y_hat)
            score_file.write(str(out)+'\n')
        
        if y == y_hat:
            A += 1
    
    print('Accuracy: ', A/len(iter))
    
    score_file.close()

    

if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]
    config_path = f'config/{dataset}.json'
    config = get_config(config_path)
    no = sys.argv[2]
    config.model_path = config.model_path + f'_{no}.pt'
    config.score_path = config.score_path + f'_{no}.txt'
    
    score_file = open(config.score_path, 'w+')    
    infer(config, score_file)
    

