import torch
import torch.nn as nn


class Sample_Concrete(nn.Module):
    def __init__(self, tau):
        super(Sample_Concrete, self).__init__()
        self.tau = tau
    def forward(self, probs):
        unif_a = torch.rand_like(probs)
        unif_b = torch.rand_like(probs)

        gumbel_a = -torch.log(-torch.log(unif_a))
        gumbel_b = -torch.log(-torch.log(unif_b))
        no_logits = (probs * torch.exp(gumbel_a))/self.tau
        de_logits = no_logits + ((1.0 - probs) * torch.exp(gumbel_b))/self.tau
        Z = no_logits / de_logits
        return Z

class Supporter(nn.Module):
  
  def __init__(self, D, C):
    super(Supporter, self).__init__()

    self.out_layer = nn.Sequential(
        nn.Linear(D, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, C)
    )

  def forward(self, x):
    out = self.out_layer(x)
    return out

class Explainer(nn.Module):
  def __init__(self, C):
    super(Explainer, self).__init__()

    self.out_layer = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, C)
    )

  def forward(self, x):
    x = x.unsqueeze(-1)
    out = self.out_layer(x)
    return out

class Selector(nn.Module):
  def __init__(self, D):
    super(Selector, self).__init__()

    self.out_layer = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
        nn.Sigmoid()

    )
  def forward(self, x):
    x = x.unsqueeze(-1)
    out = self.out_layer(x)
    out = out.squeeze(-1)
    return out

class Model(nn.Module):
  
  def __init__(self, cls, D, C):
    super(Model, self).__init__()

    self.cls = cls
 
    self.supporter = Supporter(D, C)
    self.sampler = Sample_Concrete(0.2)
    self.explainer = Explainer(C) 

    self.selector = Selector(D)

    self.out_layer = nn.Softmax(dim=1)
  
  def forward(self, x): 
    #  x: [B, D]    
    
    probs = self.selector(x) # [B, D, 1]
    if self.training: 
      Z = self.sampler(probs)
    else:
      Z = probs
    
    xz = torch.mul(x, Z)
    y2 = self.supporter(xz)

    xz_np = xz.detach().numpy()
    y1 = self.cls(xz_np)
    y1 = torch.Tensor(y1)
    
    W = self.explainer(x).permute(0, 2, 1) # [B, C, D]
    Z = Z.unsqueeze(-1)
    scores = W @ Z
    out = self.out_layer(scores).squeeze(-1)
    return out, y1, y2, W

class ModelLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1e-3):
        super(ModelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.standard_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, y):

        out = pred[0] 
        y1 = pred[1].argmax(-1) 
        y2 = pred[2]  
        W = pred[3]
        std_loss = self.standard_loss_fn(out, y1)  + self.alpha * self.standard_loss_fn(y2, y)
        loss = std_loss + self.beta * self.L21_norm(W)
        return loss

    def L21_norm(self, M):
        # sum (cols) over sum (rows)
        return torch.sum(torch.sqrt(torch.sum(M**2, dim=-1)))