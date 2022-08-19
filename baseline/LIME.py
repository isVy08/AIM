
import torch, sys
from tqdm import tqdm
import init
from utils import get_config
from lime.lime_text import LimeTextExplainer
from data_generator import DataGenerator
import numpy as np



class LIMER(object):
    def __init__(self, dg, cls, kernel_width, dataset, device):
        dg.verbose = False
        self.dg = dg
        self.cls = cls 
        self.explainer = LimeTextExplainer(kernel_width=kernel_width)
        self.device = device
        self.dataset = dataset
        

    def predict_proba(self, text_list):
        # tokenize
        inputs = self.dg._transform(text_list) 
        inputs = inputs.to(self.device)
        with torch.no_grad():
            probs = self.cls(inputs)
            if torch.cuda.is_available():
                return probs.cpu().numpy()
            return probs.detach().numpy()
            
def evaluate_lime(limer, k, num_samples, texts, labels):

    file = open(f'data/{limer.dataset}/lime_k{k}.txt', 'w+')
    
    
    N = len(texts)
    print("Test set size: ", N)
    
    for i in tqdm(range(N)):
        text = texts[i]
        y = np.argmax(labels[i], -1)
        exp = limer.explainer.explain_instance(text, limer.predict_proba, num_features=k, 
                                               num_samples=num_samples, labels=(y,))
        top_tokens = [item[0] for item in exp.as_list(label=y)]
        file.write(str(top_tokens) + '\n')
    
    file.close()
    

if __name__ == "__main__":
    dataset = sys.argv[1]
    k = int(sys.argv[2])
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    config_path = f'config/{dataset}.json'
    config = get_config(config_path)
    cls_config = get_config(config.classifier_config)
    cls_config.data_path = config.data_path
    dg = DataGenerator(cls_config)
    
    print('Loading black-box ...')
    V = dg.tokenizer.get_vocab_size()
    if cls_config.model_name == 'WordGRU':
        from blackbox import WordGRU
        cls = WordGRU(V)
        
    elif cls_config.model_name == 'WordCNN':
        from blackbox import WordCNN
        cls = WordCNN(V) 
        
    elif cls_config.model_name == 'WordTF':
        from blackbox import WordTransformer
        cls = WordTransformer(V, L = cls_config.max_length, C = cls_config.C)
    
    cls.to(device)
    

    widths = {'imdb': 15, 'hatex': 35, 'agnews': 15}
    num_samples = 2000
    
    print('Start evaluating LIME ...')
    
    kernel_width = widths[dataset]
    limer = LIMER(dg, cls, kernel_width, dataset, device)
    evaluate_lime(limer, k, num_samples, dg.test_text, dg.test_label)
