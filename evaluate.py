import string
import os
from utils import *
from utils_eval import *
from explainer import Model, Selector, Explainer
from data_generator import Tokenizer, DataGenerator


stopwords = load('./data/stopwords') + list(string.punctuation)   
wnb = load_pickle('./model/wordnet.db')

def load_base(config):
    """
    Load data generator, black-box classifer, our model and true labels
    """
    cls_config = get_config(config.classifier_config)
    cls_config.data_path = config.data_path
    
    print(f'Loading data from {cls_config.data_path} ...')
    dg = DataGenerator(cls_config)
    V = dg.tokenizer.get_vocab_size()

    # Load black-box classifier and test labels
    
    
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

    return dg, cls


def load_scores(path):

    if path is None:
        return None, None 
    
    elif 'l2x' in path:
        data = load_pickle(path)
        _, scores = data
    
    else: 
        data = load(path)
        if 'fe' in path:
            scores = [eval(item) for item in data]     
        else:
            scores = [eval(item)[0] for item in data]
    return torch.tensor(scores)

# Confidence
def log_odds(prob):
    return torch.log(prob/(1-prob)) if prob < 1.0 else torch.tensor(0.0)


def evaluate(X, cls, scores, k, device, L):
    """
    X           : input data
    cls         : black-box classifier
    scores      : feature weight vectors
    bb_labels   : predictions of black box on full data
    k           : top k features selected 
    device      : device
    L           : sequence length
    """
    
    N = X.size(0)
    A, S, B, NL, PL = 0, 0, 0, 0, 0
    

    for i in tqdm(range(N)):
        x = X[i, :].to(device)
        score = scores[i,].to(device)

        # Evaluate feature quality
        top_idx = torch.topk(score, k).indices
        top_tokens = get_top_features(x, top_idx, dg.tokenizer)
        
        B += brevity(top_tokens, wnb)
        stopwords_count = len(set(top_tokens) & set(stopwords))
        S += stopwords_count / k
  
        # Mask UNimportant words 
        neg_x = mask_data(score, x, k, 'neg', device, L)
        neg_x = neg_x.unsqueeze(0)

        # Mask IMportant words
        pos_x = mask_data(score, x, k, 'pos', device, L)
        pos_x = pos_x.unsqueeze(0)

        # Black box's predictions on full data
        prob = cls(x.unsqueeze(0))
        y = prob.argmax(-1).item()
        lo = log_odds(prob[0, y])


        # Black box's predictions on masked data
        pos_prob = cls(pos_x)
        neg_prob = cls(neg_x)

        neg_y = neg_prob.argmax(-1).item()
        pos_lo = log_odds(pos_prob[0, y])
        neg_lo = log_odds(neg_prob[0, y])

        NL += (lo - neg_lo).item()
        PL += (lo - pos_lo).item()


        if neg_y == y: 
            A += 1 
            
    return np.round(S / N, 4), np.round(B / N, 4), np.round(A / N, 4), np.round(PL / N, 4), np.round(NL / N, 4) 


if __name__ == '__main__':
    import sys
    """
    Load models and data
    ---------------------------------------------------------------------
    """
    config = get_config(sys.argv[1])
    score_path = sys.argv[2]

    scores = load_score(score_path)
    
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    dg, cls, bb_labels = load_base(config)
    print(dg.data_path)
    V = dg.tokenizer.get_vocab_size()
    L = dg.max_length

    
    # Obtain test data
    texts = dg.test_text
    X = dg._transform(texts)

    
    """
    Evaluation starts here
    ---------------------------------------------------------------------
    """
    
    k = 10
    
    S, B, A, PL, NL = evaluate(X, cls, score, k, device, L)
    print(S, B, A, PL, NL)
