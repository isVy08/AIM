import math
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
GRU for Sentiment Analysis
---------------------------------------------------------------------
"""

class WordGRU(nn.Module):
    
        
    def __init__(self, V, D = 100, L = 400, C = 2, H = 250, p = 0.1):
        super(WordGRU, self).__init__()

        self.L = L
        self.temp = 1.0
        
        self.embed_layer = nn.Embedding(V, D) # [B, F, D]
        self.gru = nn.GRU(D, H, batch_first=True, bidirectional=True)
        self.gru.flatten_parameters()

        self.out_layer = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(H, C),
        ) # [B, _, F]

   
    def forward(self, x, z = None):

        e = self.embed_layer(x) 
        if z is not None:
            e = torch.mul(e, z) 

        _, (h, c) = self.gru(e)
        g = torch.cat((h, c), dim=-1)
        
        probs = self.out_layer(g) # [B, F, 1]
        probs = torch.softmax(probs/self.temp, -1) 

        return probs
    

"""
Word CNN for Topic Classification
---------------------------------------------------------------------
"""


class WordCNN(nn.Module):

    def __init__(self, V, D = 100, C = 4, H = 250, p = 0.1):
        super().__init__()

        self.temp = 1.0
        self.L = None
        self.drop = nn.Dropout(p)
        self.emb_layer = nn.Embedding(V, D)
        widths = [10, 3, 3, 1]
        self.encoder = CNNTextLayer(D, widths = widths, filters = H)
        d_out = len(widths) * H
        self.out = nn.Sequential(
            nn.Linear(d_out, H),
            nn.ReLU(),
            nn.Linear(H, C), 
        )


    def forward(self, x, z = None):
        emb = self.emb_layer(x)
        if z is not None:
            emb = torch.mul(emb, z) 

        output = self.encoder(emb)
        output = self.drop(output)
        probs = self.out(output)
        probs = torch.softmax(probs/self.temp, -1) 
        return probs



class CNNTextLayer(nn.Module):
    def __init__(self, n_in, widths, filters=100):
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]
        x = torch.cat(x, 1)
        return x
"""
Transformer for Hate Speech Detection
---------------------------------------------------------------------
"""

class WordTransformer(nn.Module):
    def __init__(self, V, L = 200, D = 256 , C = 2, HR = 512, HL = 128):
        super().__init__()
        self.D = D
        self.L = L
        self.temp = 1.0
        self.word_emb_layer = nn.Embedding(V, D)
        self.position_emb_layer = PositionalEncoder(D, L)        
        self.transformer_layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=1024, batch_first=True) 
        self.mid_layer = nn.LSTM(D, HR, batch_first=True, bidirectional=True)
        self.mid_layer.flatten_parameters()
        self.out_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(HR * 2, HL),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HL, 64),
            nn.ReLU(),
            nn.Linear(64, C),
        
        )
       
    def forward(self, x, z=None):
    
        # broadcasting the tensor of positions 
        we = self.word_emb_layer(x) # [B, L, D]
        if z is not None:
            we = torch.mul(we, z)
        pe = self.position_emb_layer(we)

        attended = self.transformer_layer(pe) # [B, L, D]
        _, (h, _) = self.mid_layer(attended)
        out = torch.cat((h[0, :], h[1, :]), dim=-1)
        probs = self.out_layer(out)
        probs = torch.softmax(probs/self.temp, -1) 

        
        return probs


class PositionalEncoder(nn.Module):
    def __init__(self, D, L):
        super().__init__()
        self.D = D
        self.L = L
        
        pe = torch.zeros(L, D)
        for pos in range(L):
            for i in range(0, D, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/D)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/D)))
                
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        x = x * math.sqrt(self.D)
        x = x + self.pe[:,:self.L].to(x.device)
        return x