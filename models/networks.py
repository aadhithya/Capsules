import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# from keras.utils import to_categorical
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm_notebook
import pandas as pd
# from sklearn import preprocessing
import math
from collections import defaultdict

from models.layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReconNet(nn.Module):
  def __init__(self, dim=16, n_lcaps=10):
    super().__init__()
    self.fc1 = nn.Linear(dim * n_lcaps, 512)
    self.fc2 = nn.Linear(512,1024)
    self.fc3 = nn.Linear(1024,1600)
    self.dim = dim
    self.n_lcaps = n_lcaps
    self.reset_params()

  def reset_params(self):
    nn.init.normal_(self.fc1.weight, 0, 0.1)
    nn.init.normal_(self.fc2.weight, 0, 0.1)
    nn.init.normal_(self.fc3.weight, 0, 0.1)

    nn.init.constant_(self.fc1.bias,0.1)
    nn.init.constant_(self.fc2.bias,0.1)
    nn.init.constant_(self.fc3.bias,0.1)

  def forward(self, x, target):
    # mask = Variable(torch.zeros((x.size()[0],self.n_lcaps)),requires_grad=False).to(device)
    # # mask = mask.float()
    # # #import pdb; pdb.set_trace()
    # mask.scatter_(1,target.view(-1,1).long(),1.)
    # print(mask.shape)
    # mask = F.one_hot(target.long(),num_classes=self.n_lcaps)
    self.mask = target.unsqueeze(2)
    self.mask = self.mask.repeat(1,1,self.dim)

    x = x*self.mask
    x = x.view(-1,self.dim * self.n_lcaps)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


class CapsNet(nn.Module):
  def __init__(self, n_lc=10):
    super().__init__()
    self.conv1 = nn.Conv2d(1,256,kernel_size=9,stride=1)
    self.primarycaps = PrimaryCaps(256,32,8,9,2) #686 output
    self.n_primary_caps = 32*6*6
    self.digitCaps = DenseCaps_v2(nc=n_lc,num_routes=self.n_primary_caps)

    self.decoder = ReconNet()
    self.caps_score = CapsToScalars()
  def forward(self,x, target):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.primarycaps(x)

    class_caps = self.digitCaps(x)

    probs = self.caps_score(class_caps)
    
    pred = torch.argmax(probs,dim=-1)

    if target is None:
      recon = self.decoder(class_caps, pred)
    else:
      recon = self.decoder(class_caps, target)

    # Matching return statement to DeepCaps to make things a bit easier...!
    return class_caps, None, recon, pred

class DeepCaps(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv2d = nn.Conv2d(1,128,3,1,1)
    self.bn = nn.BatchNorm2d(128,1e-8,momentum=0.99)
    self.toCaps = ConvertToCaps()

    #inSize 256
    self.convcaps1_1 = ConvCaps2D(128,1,32,4,3,2,1,1) 
    self.convcaps1_2 = ConvCaps2D(32,4,32,4,3,1,1,1) #inSize 128
    self.convcaps1_3 = ConvCaps2D(32,4,32,4,3,1,1,1)
    self.convcaps1_4 = ConvCaps2D(32,4,32,4,3,1,1,1)

    #inSize 128
    self.convcaps2_1 = ConvCaps2D(32,4,32,8,3,2,1,1) 
    self.convcaps2_2 = ConvCaps2D(32,8,32,8,3,1,1,1) #inSize 64
    self.convcaps2_3 = ConvCaps2D(32,8,32,8,3,1,1,1)
    self.convcaps2_4 = ConvCaps2D(32,8,32,8,3,1,1,1)

    #inSize 64
    self.convcaps3_1 = ConvCaps2D(32,8,32,8,3,2,1,1) 
    self.convcaps3_2 = ConvCaps2D(32,8,32,8,3,1,1,1) #inSize 32
    self.convcaps3_3 = ConvCaps2D(32,8,32,8,3,1,1,1)
    self.convcaps3_4 = ConvCaps2D(32,8,32,8,3,1,1,1)

    #inSize 32
    self.convcaps4_1 = ConvCaps2D(32,8,32,8,3,2,1,1) 
    self.convcaps3d4 = ConvCaps3D(32,8,32,8,3,3) #inSize 16
    self.convcaps4_3 = ConvCaps2D(32,8,32,8,3,1,1,1)
    self.convcaps4_4 = ConvCaps2D(32,8,32,8,3,1,1,1)

    self.flat_caps = FlattenCaps()
    #numCaps for MNIST: 640
    #self, nc=10, num_routes=640, in_dim=8, out_dim=16, routing_iters=3
    self.digitCaps = DenseCaps_v2(nc=6, num_routes=8192)
    
    self.reconNet = Decoder(16,1,28,1)

    self.caps_score = CapsToScalars()
    self.mask = MaskCID()

  def forward(self, x, target=None):
    x = self.conv2d(x)
    x = self.bn(x)
    x = self.toCaps(x)

    # print(x.shape)

    #Block 1
    x =self.convcaps1_1(x)
    x_skip = self.convcaps1_2(x)
    x = self.convcaps1_3(x)
    x = self.convcaps1_4(x)
    x = x+x_skip

    #Block 2
    x =self.convcaps2_1(x)
    x_skip = self.convcaps2_2(x)
    x = self.convcaps2_3(x)
    x = self.convcaps2_4(x)
    x = x+x_skip

    #Block 3
    x =self.convcaps3_1(x)
    x_skip = self.convcaps3_2(x)
    x = self.convcaps3_3(x)
    x = self.convcaps3_4(x)
    x = x+x_skip
    x1 = x

    #Block 1
    x =self.convcaps4_1(x)
    x_skip = self.convcaps3d4(x)
    x = self.convcaps3_3(x)
    x = self.convcaps3_4(x)
    x = x+x_skip
    x2 = x

    xa = self.flat_caps(x1) # 512 Capsules
    xb = self.flat_caps(x2) # 128 Capsules

    x = torch.cat([xa,xb],dim=-2)

    class_caps = self.digitCaps(x)
    x = self.caps_score(class_caps)
    masked, indices = self.mask(class_caps, target)
    decoded = self.reconNet(masked)

    return class_caps, masked, decoded, indices
