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
    f = x[0, top_idx] 
    f = f[f > 0]
    top_tokens = tokenizer.decode(f.tolist())
    return top_tokens

## MASK DATA
def mask_data(score, x, k, mask_type, device, L):

    top_idx = torch.topk(score, k).indices
    if mask_type == 'neg': # mask negative words, leave important words
        selected = top_idx.tolist()
    else: # mask positive words
        selected = [j for j in range(L) if j not in top_idx]
                
    selected_x = x[selected]
    masked_x = torch.cat((selected_x, torch.zeros(L-len(selected_x), device=device).long())) 
    
    return masked_x 


## BREVITY
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
    return len(clusters) # the smaller, the better
