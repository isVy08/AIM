import string
import os
from utils import *
from utils_eval import *
from lime.lime_text import LimeTextExplainer
from explainer import Model, Selector, Explainer
from data_generator import Tokenizer, DataGenerator
from baseline.LIME import LIMER, evaluate_lime


stopwords = load('./data/stopwords') + list(string.punctuation)   
wnb = load_pickle('./data/wordnet.db')

def load_base(config):
    """
    Load data generator, black-box classifer, our model and true labels
    """
    cls_config = get_config(config.classifier_config)

    dg = DataGenerator(cls_config)
    V = dg.tokenizer.get_vocab_size()

    # Load black-box classifier and test labels
    bb_data = load_pickle(config.data_path)
    val_label = bb_data['val'][1]
    k = int(0.2 * len(val_label))

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
    cls.temp = config.temp
    cls.eval()
    
    bb_labels = torch.Tensor(val_label[-k:]).argmax(-1)

    print(len(bb_labels))

    return dg, cls, bb_labels


def load_outputs(path):

    if path is None:
        return None, None 
    
    elif 'l2x' in path:
        data = load_pickle(path)
        preds, scores = data
        preds = preds.argmax(-1)
    
    else: 
        data = load(path)
        scores = [eval(item)[0] for item in data]
        preds = [eval(item)[1] for item in data]
    
    return preds, torch.tensor(scores)


def evaluate(X, cls, scores, bb_labels, labels, k, device, L):
    """
    X           : input data
    cls         : black-box classifier
    scores      : feature weight vectors
    bb_labels   : predictions of black box on full data
    labels      : ground-truth label
    k           : top k features selected 
    device      : device
    L           : sequence length
    """
    
    N = X.size(0)
    A, S, B = 0, 0, 0
    

    for i in tqdm(range(N)):
        x = X[i, :].to(device)
        score = scores[i,].to(device)

        # Evaluate feature quality
        top_idx = torch.topk(score, k).indices
        top_tokens = get_top_features(x, top_idx, tokenizer)
        
        B += brevity(top_tokens, wnb)
        stopwords_count = len(set(top_tokens) & set(stopwords))
        S += stopwords_count / k
  
        # Mask UNimportant words 
        masked_x = mask_data(score, x, k, 'neg', device, L)
        masked_x = masked_x.unsqueeze(0)

        # Black box's predictions on masked data
        bb_pred = cls(masked_x, None)
        bb_y = bb_pred.argmax(-1).item()

        if bb_y == bb_labels[i].item(): 
            A += 1 
            
    return round(A / N, 4), round(S / N, 4), round(B / N, 4)


if __name__ == '__main__':
    import sys
    """
    Load models and data
    ---------------------------------------------------------------------
    """
    config = get_config(sys.argv[1])
    model_name = sys.argv[2] # AIM, L2X, VIBI or LIME
    
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    dg, cls, bb_labels = load_base(config)
    print(dg.data_path)
    V = dg.tokenizer.get_vocab_size()
    L = dg.max_length



    # Load baseline
    if model_name != 'LIME':
        path = sys.argv[3] # path to weight vector
        _, score = load_outputs(path)
        
    elif model_name == 'LIME':
        kernel_width = 15
        limer = LIMER(dg, cls, kernel_width, device)
            
    
    # Obtain test data
    texts = dg.test_text
    labels = dg.test_label

    X = dg._transform(texts)

    
    """
    Evaluation starts here
    ---------------------------------------------------------------------
    """
    
    k = 10
    

    if model_name in ('L2X', 'VIBI', 'AIM'):
        
        A, S, N = evaluate(X, cls, score, bb_labels, labels, k, device, L)
        print(f"Faithfulness: {A:.3f} - Purity: {S:.3f} - Brevity: {B:.3f}")

    elif model_name == 'LIME':
        num_samples = 4000
    
        A, S, B = evaluate_lime(limer, num_samples, texts, bb_labels, k, wnb, stopwords)
        print(f"Faithfulness: {A:.3f} - Purity: {S:.3f} - Brevity: {B:.3f}")
    
    else: 
        raise ValueError("Please specify AIM, L2X, VIBI or LIME ")