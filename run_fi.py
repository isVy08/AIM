import numpy as np
from tqdm import tqdm
import os
import sys
from utils import *

from fi_code.fi import *
from fi_code.utils_fe import *
from fi_code.basic_models import *


from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from evaluate import load_base
from data_generator import Tokenizer, DataGenerator

 

name = sys.argv[1]
model_no = sys.argv[2]

config = get_config(f'config/{name}.json')
config.device = 'cpu'

data = load_pickle(config.data_path)
train_data, train_label = data['train']
val_data, val_label = data['val']
val_label = np.argmax(val_label, -1)
train_label = np.argmax(train_label, -1)
size = 100
test_data, test_label = val_data[-size:], val_label[-size:]




dg, cls, bb_labels = load_base(config)






def create_preprocess_function(dg):
    def preprocess_function(examples):
        encoded = dg._transform(examples['text'])
        tokenized = {'input_ids': encoded.tolist()}
        
        return tokenized
    return preprocess_function

def prepare_data_loaders(dg, batch_size: int = 128):

    
    d = {'train':Dataset.from_dict({'label':train_label,'text':train_data}),
     'test':Dataset.from_dict({'label':val_label,'text':val_data})}
    dataset = DatasetDict(d)

    ppf = create_preprocess_function(dg)
    tokenized = dataset.map(ppf, batched=True)
    tokenized['train'].set_format(type='torch', columns=['input_ids', 'label'])
    tokenized['test'].set_format(type='torch', columns=['input_ids', 'label'])
    train_loader = DataLoader(tokenized['train'], batch_size=batch_size, num_workers=1, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(tokenized['test'], batch_size=batch_size, num_workers=1)
    return train_loader, test_loader


train_dataloader, test_dataloader = prepare_data_loaders(dg, 128) # change dataset here


   
labels_dict = {0: 'Negative', 1: 'Positive'}

def run(texts, labels, n=10):
    tokenized = create_preprocess_function(dg)({'text': texts})
    # tokens = batch_decode(tokenized['input_ids'], tokenizer)
    embeddings = torch.tensor(extract_embeddings(tokenized['input_ids'], cls, device='cpu'))  # this is the input we will explain
    inputs = list(embeddings)
    # attention_mask = torch.tensor(tokenized['attention_mask'])
    covariances, choleskies = calculate_text_covariances(train_dataloader, cls)
    inputs_covs = [covariances[i] for i in labels]
    inputs_chols = [choleskies[i] for i in labels]

    torch.backends.cudnn.enabled = False
    activation = torch.nn.Softmax(dim=-1)
    methods = ['functional_information']
    explanations = explain_batch(cls, inputs, methods,
                                 modality='text', n=n, label=labels, pertubation_device='cpu',
                                 covariance=inputs_covs, cholesky_decomposed_covariance=inputs_chols,
                                 outputs_activation_func=activation)
    return explanations[0]

def combine(input_ids, tokens):
    L = len(input_ids)
    words = []
    stacked_ids = []
    for j in range(L):
        tok = tokens[j]
        if not tok.startswith('##'):
            words.append(tok)
            stacked_ids.append([j])
        else:
            tok = tok.replace('##', '')
            words[-1] = words[-1] + tok
            stacked_ids[-1].append(j)
    return words, stacked_ids

def get_top_words(words, stacked_ids, top_indices):
    lw = len(words)
    top_words = []
    for j in range(lw):
        idx = stacked_ids[j]
        items = [p for p in idx if p in top_indices]
        if len(items) > 0:
            top_words.append(words[j])
    return top_words

dg.verbose = False

f = open(f'data/fe_imdb/fe_score{model_no}.txt','w+')
for i in tqdm(range(size)):
    texts = [test_data[i]]
    labels = [int(test_label[i])]
    score = run(texts, labels, 20)
    f.write(str(score.tolist()) + '\n')

f.close()