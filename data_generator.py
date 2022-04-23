import torch, os, re
import numpy as np
from utils import *
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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
        self.text_level = config.text_level
        self.path_to_tokenizer = config.path_to_tokenizer
        self.C = config.C
        self.verbose = True

        self.model_name = config.model_name
        
        if os.path.isfile(config.data_path):
            data = load_pickle(config.data_path)
            self.train_text, self.train_label = data["train"]
            val_text, val_label = data["val"]
            
            k = int(0.2 * len(val_text))
            self.val_text = val_text[:-k]
            self.val_label = val_label[:-k]
            self.test_text = val_text[-k:]
            self.test_label = val_label[-k:]
            
    
        elif 'IMDB' in config.data_path:
            from torchtext.datasets import IMDB
            # 0: negative, 1: positive
            print('Reading IMDB dataset ...')
            train_iter = IMDB(split='train')
            self.train_text, self.train_label = self._load(train_iter)
            val_iter = IMDB(split='test')
            self.val_text, self.val_label = self._load(val_iter)
            data = {"train": (self.train_text, self.train_label), "val": (self.val_text, self.val_label)}
            write_pickle(data, config.data_path)
        
        elif 'AG' in config.data_path:
            from torchtext.datasets import AG_NEWS
            print('Reading AG-NEWS dataset ...')
            # World (0), Sports (1), Business (2), Sci/Tech (3)
            train_iter = AG_NEWS(split='train')
            self.train_text, self.train_label = self._load(train_iter)
            val_iter = AG_NEWS(split='test')
            self.val_text, self.val_label = self._load(val_iter)
            data = {"train": (self.train_text, self.train_label), "val": (self.val_text, self.val_label)}
            write_pickle(data, config.data_path)
        
        elif 'HATEX' in config.data_path:
            print('Reading HateXplain dataset ...')
            import statistics as stats
            # 0: normal, 1: hatespeech, 2: offensive
            orig_data = load_json('data/hatexplain.json')
            splits = load_json('data/hatexplain_split.json')
            label_set = {'normal': 0, 'hatespeech': 1, 'offensive': 1}

            self.train_text, self.train_label = [], []
            self.val_text, self.val_label = [], []
            t = 0
            for k, v in tqdm(orig_data.items()):
                tokens = v['post_tokens']
                ants = [label_set[item['label']] for item in v['annotators']]
                try:
                    l = stats.mode(ants)
                except stats.StatisticsError:
                    l = max(ants)
                if k in splits['train'] and t < 15000: 
                    self.train_text.append(' '.join(tokens) + ' .')
                    self.train_label.append(l)
                    t += 1
                else:
                    self.val_text.append(' '.join(tokens) + ' .')
                    self.val_label.append(l)
                
            data = {"train": (self.train_text, self.train_label), "val": (self.val_text, self.val_label)}
            write_pickle(data, config.data_path)

        if self.text_level == 'sent':
            self.max_sentence_length = config.max_sentence_length

        
        # Tokenizer:

        if os.path.isfile(self.path_to_tokenizer):
            self.tokenizer = load_pickle(self.path_to_tokenizer)
                
        else: 
            texts = []
            for dataset_name in ('IMDB', 'AG', 'HATEX'):
                data = load_pickle(f'data/{dataset_name}.pickle')
                train_text, _ = data["train"]
                val_text, _ = data["val"]
                texts.extend(train_text)
                texts.extend(val_text)

            self.tokenizer = train_tokenizer(texts, 
                                             self.min_frequency, 
                                             self.path_to_tokenizer)
        
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx = 2, 3, 0, 1

    
    def generate_data(self):
        
        if self.data_fraction < 1.0:
            n = len(self.train_text)
            k = int(self.data_fraction * n)
            self.train_text = self.train_text[:k]
            self.train_label = self.train_label[:k]
            self.val_text = self.val_text[:k]
            self.val_label = self.val_label[:k]
        

        self.train_x, self.val_x = self._transform(self.train_text),  self._transform(self.val_text)
        self.test_x = self._transform(self.test_text)
        
        if self.model_name == 'L2X':
                
            self.train_y, self.val_y = np.array(self.train_label), np.array(self.val_label)
            self.test_y = np.array(self.test_label)
        else:
            self.train_y, self.val_y = torch.Tensor(self.train_label), torch.Tensor(self.val_label)
            self.test_y = torch.Tensor(self.test_label)

    
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
            if self.text_level == 'word':
                text = re.sub('<[^<]+?>', '', text)
                ids = self.tokenizer.encode(text)
                ids = self._pad_trunc(ids, self.max_length)
                input.append(ids)
            elif self.text_level == 'sent': 
                sent_input = []
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

        if self.model_name == 'L2X':
            input = np.array(input)
        else:
            input = torch.LongTensor(input) 
        if self.verbose:
            print('Finish tokenization!')
        return input
