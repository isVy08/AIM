import torch
import torch.nn as nn
import torch.nn.functional as F


class Explainer(nn.Module):
    
    def __init__(self, V, D, C, H):
        super(Explainer, self).__init__()

        
        
        self.embed_layer = nn.Embedding(V, D) # [B, L, D]
        self.linear_layer = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, C),
        ) 
        
        

    def forward(self, x):

        e = self.embed_layer(x) 
        W = self.linear_layer(e)
        return torch.swapaxes(W, 1, 2)

class Selector(nn.Module):
    
    def __init__(self, V, D, L, H, p):
        super(Selector, self).__init__()

        self.L = L
        
        self.emb_layer = nn.Embedding(V, D) # [B, L, D]
        self.lstm = nn.LSTM(D, H, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

        self.out_layer = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(H * 2, H),
            nn.ReLU(),

            nn.Dropout(p),
            nn.Linear(H, H),
            nn.ReLU(),
            
            nn.Dropout(p),
            nn.Linear(H, 1),
            nn.Sigmoid()
        ) # [B, _, F]

   
    def forward(self, x):

        e = self.emb_layer(x)  

        h, _ = self.lstm(e)
        
        probs = self.out_layer(h) # [B, F, 1]

        return probs
        
     
class Sample_Concrete(nn.Module):
    def __init__(self, tau):
        super(Sample_Concrete, self).__init__()
        self.tau = tau
    def forward(self, probs):
        unif_a = torch.rand(probs.shape)
        unif_b = torch.rand(probs.shape)

        gumbel_a = -torch.log(-torch.log(unif_a))
        gumbel_b = -torch.log(-torch.log(unif_b))
        no_logits = (probs * torch.exp(gumbel_a))/self.tau
        de_logits = no_logits + ((1.0 - probs) * torch.exp(gumbel_b))/self.tau
        Z = no_logits / de_logits
        return Z


class Approximator(nn.Module):
    
        
    def __init__(self, V, D, L, C, H, in_kernel, p):
        super(Approximator, self).__init__()

        self.L = L
        
        self.emb_layer = nn.Embedding(V, D) # [B, F, D]
        self.in_layer = nn.Sequential(
            nn.Dropout(p),
            nn.Conv1d(D, H, in_kernel, padding='same', stride=1),
            nn.ReLU()
        )

        self.pool_layer = nn.MaxPool1d(L) # [B, H, 1]

        self.out_layer = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(H, C),
            nn.Softmax(-1)
        ) # [B, _, F]

   
    def forward(self, x, z):

        e = self.emb_layer(x)
        e = torch.mul(e, z)  
        e = torch.swapaxes(e, 1, 2)
             
        e = self.in_layer(e)
        
        # Global info
        g = self.pool_layer(e)

        probs = self.out_layer(g.squeeze(-1)) # [B, C]

        return probs


class Model(nn.Module):
    def __init__(self, V, D, L, C, HS, HE, HU, in_kernel, device, p, tau, cls):
        super(Model, self).__init__()
        
        self.device = device
        self.tau = tau
        self.L = L


        self.explainer = Explainer(V, D, C, HE)
        self.selector = Selector(V, D, L, HS, p)
        self.sampler = Sample_Concrete(tau)
        self.cls = cls
        self.sup = Approximator(V, D, L, C, HU, in_kernel, p)
        
        self.out_layer = nn.Softmax(dim=1)

        

    def forward(self, x):
        
        
        W = self.explainer(x) # [B, C, L]
        probs = self.selector(x)

        # Sample bernoulli
            
        if self.training:
            Z = self.sampler(probs) # [B, L, 1]
        else:
            Z = probs
                    
        for params in self.cls.parameters():
            params.requires_grad = False
            
        y1 = self.cls(x, Z).unsqueeze(-1) # [B, L, 1]
        scores = W @ Z # [B, C, 1]
        
        out = self.out_layer(scores)
        y2 = self.sup(x, Z).unsqueeze(-1)    
        return torch.cat((out, y1, y2, W), dim=-1)
       

        
        
        
        
        
        
