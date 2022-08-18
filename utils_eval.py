import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from lemminflect import getAllLemmas
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from utils import load, load_pickle


## GET TOP TOKENS
def get_top_features(x, top_idx, tokenizer):
    f = x[top_idx] 
    f = f[f > 0]
    top_tokens = tokenizer.decode(f.tolist())
    return top_tokens

## MASK DATA
def mask_data(score, x, k, mask_type, device, L):

    top_idx = torch.topk(score, k).indices
    if mask_type == 'neg': # mask negative words, retain important words
        selected = top_idx.tolist()
    else: # mask positive words, retain unimportant words
        selected = [j for j in range(L) if j not in top_idx]
                
    selected_x = x[selected]
    masked_x = torch.cat((selected_x, torch.zeros(L-len(selected_x), device=device).long())) 
    
    return masked_x 


## BREVITY: smaller is better
def brevity(top_tokens, wnb):    
    synonyms = {}
    for token in top_tokens:
        if token not in synonyms:
            synonyms[token] = set()
        lemma = getAllLemmas(token).values()
        lemma = list(set(chain(*lemma)))
        for word in lemma:
            try:
                family = wnb[word]
                for mem in family:
                    syns = set(mem['synonyms'] + mem['related'] + ['similar'])
                    synonyms[token].update(syns)
            except:
                pass

    clusters = {}
    keys = list(synonyms.keys())
    N = len(keys)
    for i in range(N):
        orig = keys[i]
        if orig in synonyms:
            clusters[orig] = []
            for j in range(i+1, N):
                ref = keys[j]
                if ref in synonyms:
                    if len(synonyms[orig] & synonyms[ref]) > 0:
                        clusters[orig].append(ref)
                        del synonyms[ref]
    return len(clusters) 


# STABILITY (semantics similarity): higher is better
def convert_to_bert_hidden_states(tokenizer, model, x):
    with torch.no_grad():
        inputs = tokenizer(x, return_tensors="pt", max_length=400, truncation=True)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs['hidden_states'][-1]
        return torch.sum(hidden_states.squeeze(0), dim=0)

def bert_cos_sim(a, b, dummy):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=1, eps=1e-6)
    return cos(a, b).item()