import torch
import torch.nn as nn

"""
 Black-box Model Architecture

"""

class Net(nn.Module):
    def __init__(self, C = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, C)
        self.out = nn.Softmax(-1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.out(x)



"""
 Model Explainer Architecture

"""
class Patch(nn.Module):
    
    def __init__(self):
        super(Patch, self).__init__()

    def forward(self, x):
      patches = []
      for i in range(0, 28, 7): # t b l r
        for j in range(0, 28, 7):
          patches.append(x[:, :, i:i+7, j:j+7])
      x = torch.stack(patches, 1)
      return x.view(-1, 1, 7, 7)

class Merge(nn.Module):
    
    def __init__(self):
        super(Merge, self).__init__()

    def forward(self, e):
      patches = []
      for start, end in ((0, 4), (4, 8), (8, 12), (12, 16)):  
        patch = [e[:, i, :, :] for i in range(start, end)]
        patches.append(torch.cat(patch, dim=-1))  
      img = torch.cat(patches, dim=1)
      return img.unsqueeze(1)

class Supporter(nn.Module):
  
  def __init__(self, C):
    super(Supporter, self).__init__()

    self.in_layer = nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(1, 32, kernel_size = 3, padding = 'same', stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2)),
        nn.Dropout(0.2),
        nn.Conv2d(32, 64, kernel_size = (3, 3), padding = 'same', stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2))
    )

    self.out_layer = nn.Sequential(
        nn.Linear(64 * 7 * 7, C),
        nn.Softmax(1)
    )

  def forward(self, x):
    x = self.in_layer(x)
    x = x.view(-1, 64 * 7 * 7) 
    out = self.out_layer(x)
    return out

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

class Explainer(nn.Module):
  def __init__(self, C):
    super(Explainer, self).__init__()
    self.in_layer = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size = 3, padding = 'same', stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2)),
        nn.Conv2d(32, 64, kernel_size = 3, padding = 'same', stride=1),  
        nn.ReLU(),    
        nn.MaxPool2d(kernel_size = (3, 3))      
    )

    self.out_layer = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, C)
    )

  def forward(self, x):
    x = self.in_layer(x)
    x = x.view(-1, 16, 64)
    out = self.out_layer(x)
    return out


class Selector(nn.Module):
  def __init__(self):
    super(Selector, self).__init__()
    self.in_layer = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size = 3, padding = 'same', stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2, 2)),
        nn.Conv2d(32, 64, kernel_size = 3, padding = 'same', stride=1),  
        nn.ReLU(),    
        nn.MaxPool2d(kernel_size = (3, 3))
    )

    self.out_layer = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()

    )
  def forward(self, x):
    x = self.in_layer(x) # [b * 16, 32, 1, 1]
    x = x.view(-1, 16, 64)
    out = self.out_layer(x)
    return out

class Multiply(nn.Module):
    
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, e, z):
      e = e.view(-1, 16, 7, 7)
      z = z.unsqueeze(-1)
      out = torch.mul(e, z) # [b, 16, 7, 7]
      out = Merge()(out)
      return out

class Model(nn.Module):
  
  def __init__(self, cls, C):
    super(Model, self).__init__()

    self.cls = cls
    self.supporter = Supporter(C)
    self.sampler = Sample_Concrete(0.2)
    self.explainer = Explainer(C) 

    self.selector = Selector()

    self.out_layer = nn.Softmax(dim=1)
  
  
  
  def forward(self, x): 
    e =  Patch()(x) # [b * 4, 1, 7, 7]
    W = self.explainer(e).permute(0, 2, 1) # [B, C, 4]
    probs = self.selector(e) # [B, 4, 1]
    if self.training: 
      Z = self.sampler(probs)
    else:
      Z = probs
    
    
    xp = Multiply()(e, probs)
    y1 = self.cls(xp)
    
    scores = W @ probs
    out = self.out_layer(scores).squeeze(-1)

    xz = Multiply()(e, Z)
    y2 = self.supporter(xz)
    return out, y1, y2, W