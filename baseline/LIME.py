
import string
import torch
from tqdm import tqdm
from utils import load
from utils_eval import *
from lime.lime_text import LimeTextExplainer

class LIMER(object):
    def __init__(self, dg, cls, device):
        dg.verbose = False
        self.dg = dg
        self.cls = cls 
        self.explainer = LimeTextExplainer()
        self.device = device
        

    def predict_proba(self, text_list):
        # tokenize
        inputs = self.dg._transform(text_list) 
        inputs = inputs.to(self.device)
        with torch.no_grad():
            probs = self.cls(inputs)
            if torch.cuda.is_available():
                return probs.cpu().numpy()
            return probs.detach().numpy()
            
def evaluate_lime(limer, num_samples, 
                texts, labels, bb_labels, k, 
                 wnb, stopwords):
    
    
    N = len(texts)

    A, S, B = 0, 0, 0
    labs = tuple(range(limer.dg.C))

    for i in tqdm(range(N)):
        text = texts[i]
        exp = limer.explainer.explain_instance(text, limer.predict_proba, num_features=k, num_samples=num_samples, labels=labs) # explain label 1
        top_tokens = [item[0] for item in exp.as_list()]

        # Evaluate quality
        B += brevity(top_tokens, wnb)
        # Stop words ratio
        stopwords_count = len(set(top_tokens) & set(stopwords))
        S += stopwords_count / k
        
        
        # get X ids --> obtain token ids --> masked
        masked_text = ' '.join(top_tokens)
        masked_input = limer.dg._transform([masked_text])
        masked_input = masked_input.to(limer.device)
        bb_y = limer.cls(masked_input).argmax(-1).item()

        if bb_y == bb_labels[i].item():
            A += 1 
    

    return round(A / N, 4), round(S / N, 4), round(B / N, 4)
  