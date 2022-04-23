import os
import string
import torch
from utils import *
from tqdm import tqdm
from utils_eval import *
from explainer import Model, Selector, Explainer
from data_generator import Tokenizer, DataGenerator

stopwords = load('./data/stopwords') + list(string.punctuation)   
wnb = load_pickle('./data/wordnet.db')


def infer(config, k, output_file, score_file): 
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
    A, B, S = 0, 0, 0
    
    for i in tqdm(iter):

        text = dg.test_text[i]
        y = np.argmax(dg.test_label[i])

        x = test_x[i, :].to(device)
        x = x.unsqueeze(0)

        pred = model(x)
        y_hat = pred[:, :, 0].argmax().item() # explainer's predictions
        weight = pred[0, y, 3:] # spatial weight W

        # Write weight vectors for conventional inference
        if score_file is not None:
            out = (weight.tolist(), y_hat)
            score_file.write(str(out)+'\n')


        ##### Adaptive inference begins #####
        
        _, indices = torch.sort(weight, descending=True)
        for j in range(L):
            masked_x = torch.zeros_like(x)
            masked_x[:, :j+1] = x[:, indices[:j+1]]
            bb_y = cls(masked_x, None).argmax(-1).item()
            if bb_y == y:
                A += 1
                break 
                
            elif j >= k:
                break
        
        
        idx = indices[:j+1]        
        tokens = get_top_features(x, idx, dg.tokenizer)
        
        B += brevity(tokens, wnb)
        # Stop words ratio
        stopwords_count = len(set(tokens) & set(stopwords))
        S += stopwords_count / (j+1)


        # Write top k tokens 
        top_k = indices[:k]        
        top_k_tokens = get_top_features(x, idx, dg.tokenizer)

        if output_file is not None:
            content = f"{i}. {text}\n\n{j+1} optimal feature(s). Top {k} features: {tokens}\n\nExplainer's label: {y_hat} - Black-box's label: {y}\n"
            output_file.write(content)
            output_file.write('*'*10+'\n')
    
    N = len(iter)
    if score_file is not None:
        score_file.close()
    
    if output_file is not None:
        output_file.close()
    return round(A / N, 4), round(S / N, 4), round(B / N, 4)
    

if __name__ == "__main__":
    import sys
    config = get_config(sys.argv[1]) 
    k = 10 # replace 10 with whatever value you want to investigate
    output_file = open(config.output_path, 'w+')
    score_file = open(config.score_path, 'w+')    
    A, S, B = infer(config, k, output_file, score_file)
    print(f"Faithfulness: {A:.3f} - Purity: {S:.3f} - Brevity: {B:.3f}")

