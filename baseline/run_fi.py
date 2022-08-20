import numpy as np
from tqdm import tqdm
import os
import sys

from fi import *
from utils_fi import *
from basic_models import *


import init
from utils import *
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from evaluate import load_base
from data_generator import Tokenizer, DataGenerator

 

dataset = sys.argv[1]

config = get_config(f'config/{dataset}.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dg, cls = load_base(config)

size = 100
test_text, test_label = dg.test_text[:size], np.argmax(dg.test_label[:size],-1)
# test_text, test_label = dg.test_text[size:], np.argmax(dg.test_label[size:],-1)






def create_preprocess_function(dg):
    def preprocess_function(examples):
        encoded = dg._transform(examples['text'])
        tokenized = {'input_ids': encoded.tolist()}
        
        return tokenized
    return preprocess_function

def prepare_data_loaders(dg, batch_size: int = 128):

    train_label = np.argmax(dg.train_label, -1)
    val_label = np.argmax(dg.val_label, -1)
    d = {'train':Dataset.from_dict({'label':train_label,'text':dg.train_text}),
     'test':Dataset.from_dict({'label':test_label,'text':test_text})}
    dataset = DatasetDict(d)

    ppf = create_preprocess_function(dg)
    tokenized = dataset.map(ppf, batched=True)
    tokenized['train'].set_format(type='torch', columns=['input_ids', 'label'])
    tokenized['test'].set_format(type='torch', columns=['input_ids', 'label'])
    # train_loader = DataLoader(tokenized['train'], batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(tokenized['test'], batch_size=batch_size, num_workers=1)
    return test_loader


test_dataloader = prepare_data_loaders(dg, 128) # change dataset here


   
labels_dict = {0: 'Negative', 1: 'Positive'}

def run(texts, labels, k=100):
    tokenized = create_preprocess_function(dg)({'text': texts})
    # tokens = batch_decode(tokenized['input_ids'], tokenizer)
    embeddings = torch.tensor(extract_embeddings(tokenized['input_ids'], cls, device=device))  # this is the input we will explain
    inputs = list(embeddings)
    # attention_mask = torch.tensor(tokenized['attention_mask'])
    covariances, choleskies = calculate_text_covariances(test_dataloader, cls)
    inputs_covs = [covariances[i] for i in labels]
    inputs_chols = [choleskies[i] for i in labels]

    torch.backends.cudnn.enabled = False
    activation = torch.nn.Softmax(dim=-1)
    methods = ['covariance_functional_information']
    explanations = explain_batch(cls, inputs, methods,
                                 modality='text', n=k, label=labels, pertubation_device=device,
                                 covariance=inputs_covs, cholesky_decomposed_covariance=inputs_chols,
                                 outputs_activation_func=activation)
    return explanations[0]

dg.verbose = False
num_samples = 200

f = open(f'data/{dataset}/fe_{num_samples}.txt','w+')
for i in tqdm(range(size)):
    texts = [test_text[i]]
    labels = [int(test_label[i])]
    score = run(texts, labels, num_samples)
    f.write(str(score.tolist()) + '\n')

f.close()