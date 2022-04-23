import torch, os, re
import numpy as np
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from utils import load_pickle, write_pickle, load_json

class Tokenizer(object):
    
    def __init__(self, spacy_tokenizer, torch_vocab):
        self.vocab = torch_vocab
        self.tokenizer = spacy_tokenizer
    
    def encode(self, sequence):
        tokens_list = self.tokenizer(sequence)
        return self.vocab(tokens_list)
    
    def decode(self, indices):
        return self.vocab.lookup_tokens(indices)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def __str__(self):
        return self.__class__.__name__

def train_tokenizer(data_iter, min_frequency, path_to_tokenizer):
    print('Training tokenizer ...')
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"] 
    
    spacy_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    
    data = []
    for line in tqdm(data_iter):
        line = line.strip().lower()
        data.append(spacy_tokenizer(line))
    
    vocab = build_vocab_from_iterator(data, min_freq=min_frequency, 
                                        specials=special_tokens, special_first=True)
    
    vocab.set_default_index(1)
    tokenizer = Tokenizer(spacy_tokenizer, vocab)
    
    print('Saving tokenizer ...')
    write_pickle(tokenizer, path_to_tokenizer)
        
    return tokenizer

class DataGenerator(object):
    

    def __init__(self, config):
        super(DataGenerator).__init__()

        self.data_path = config.data_path
        self.min_frequency = config.min_frequency
        self.max_length = config.max_length
        self.data_fraction = config.data_fraction
        self.path_to_tokenizer = config.path_to_tokenizer
        self.C = config.C
        self.verbose = True

        self.max_sentence_length = config.max_sentence_length
        
        # Load data
        data = load_pickle(config.data_path)
        self.train_text, self.train_label = data["train"]
        val_text, val_label = data["val"]
        
        k = int(0.2 * len(val_text))
        self.val_text = val_text[:-k]
        self.val_label = val_label[:-k]
        self.test_text = val_text[-k:]
        self.test_label = val_label[-k:]
            
        # Load tokenizer
        
        self.tokenizer = load_pickle(self.path_to_tokenizer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx = 2, 3, 0, 1


    
    def generate_data(self):
        
        if self.data_fraction < 1.0:
            n = len(self.train_text)
            k = int(self.data_fraction * n)
            self.train_text = self.train_text[:k]
            self.train_label = self.train_label[:k]        

        self.train_x, self.val_x = self._transform(self.train_text),  self._transform(self.val_text)
        self.test_x = self._transform(self.test_text)
        self.train_y, self.val_y = torch.LongTensor(self.train_label), torch.LongTensor(self.val_label)
        self.test_y = torch.LongTensor(self.test_label)

    
    def _load(self, data_iter):
        print("Loading data ...")
        X, Y = [], []
        for label, text in data_iter:
            if 'IMDB' in self.data_path:
                y = 1.0 if label == 'pos' else 0.0
            elif 'AG' in self.data_path:
                y = label - 1
            Y.append(y)
            text = self._preprocess(text)
            X.append(text)
        return X, Y

    def _preprocess(self, text):
        text = re.sub('<[^<]+?>', '', text)
        return text.strip().lower()
    
    
    def _pad_trunc(self, ids, max_length):
        if len(ids) > max_length:
            ids = ids[: max_length]
        elif len(ids) < max_length:
            pad_size = max_length - len(ids)
            ids = ids + [self.pad_idx] * pad_size 
        return ids


    def _transform(self, data):
        
        input = []
        if self.verbose:
            iter = tqdm(data)
            print('Tokenizing data ...')
        else:
            iter = data
        for text in iter:
            sent_input = []
            sents = text.split('. ')
            if len(sents) > self.max_length:
                sents = sents[:self.max_length]
            for sent in sents:
                ids = self.tokenizer.encode(sent)
                ids = self._pad_trunc(ids, self.max_sentence_length)
                sent_input.append(ids) 
            if len(sents) < self.max_length:
                pad_size = self.max_length - len(sents)
                for _ in range(pad_size):
                    sent_input.append([self.pad_idx] * self.max_sentence_length)
            input.append(sent_input)

        input = torch.LongTensor(input) 
        if self.verbose:
            print('Finish tokenization!')
        return input
