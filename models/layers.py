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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def squash(s, dim=-1):
  eps=1e-8
  norm = torch.norm(s, dim=dim, keepdim=True)
  return (norm /(1 + norm**2 + eps)) * s

def softmax3D(x, dim):
  return (torch.exp(x) / torch.sum(torch.sum(torch.sum(torch.exp(x), dim=dim[0], keepdim=True), dim=dim[1], keepdim=True), dim=dim[2], keepdim=True))

def one_hot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor.cuda()) # One-hot encode

class ConvertToCaps(nn.Module):
  def __init__(self):
    '''
    Convert ConvLayer Outputs into Capsules.
    ConvLayer Out: NxCxHxW
    Capsule Shape: NxNCxDxHxW
    NC: Number of capsules.
    D: Capsule Dimension.
    '''
    super().__init__()

  def forward(self, x):
    return x.unsqueeze(2)

class FlattenCaps(nn.Module):
  def __init__(self):
    '''
    Transposes and Flattens capsules.
    Input Shape: NxNCxDxHxW
    Output Shape: Nx[NCxHxW]xD
    '''
    super().__init__()
  def forward(self, x):
    n, nc, d, h, w = x.shape
    x = x.permute(0,3,4,1,2).contiguous()
    return x.view(n,nc*h*w,d)

class CapsToScalars(nn.Module):
  def __init__(self):
    '''
    Returns norm of capsule taken along the capsule dimension dim.
    Norm/Length of capsules is the probablity that the 
    object detected by that capsule exists.
    '''
    super().__init__()
  
  def forward(self, x):
    return torch.norm(x,dim=2)

class DenseCaps_v2(nn.Module):
  def __init__(self, nc=10, num_routes=640, in_dim=8, out_dim=16, routing_iters=3):
      '''
      Dense Capsule Layer.
      '''
      super().__init__()
      self.nc = nc
      self.num_routes = num_routes
      self.out_dim=out_dim
      self.r_it = routing_iters

      self.W = nn.Parameter(torch.Tensor(num_routes, in_dim, nc * out_dim))
      self.bias = nn.Parameter(torch.rand(1, 1, nc, out_dim) * 0.01)
      self.b = nn.Parameter(torch.zeros(num_routes,nc))
      self.reset_params()
  def reset_params(self):
    # stdv = 1/math.sqrt(self.num_routes)
    self.W.data.normal_(0,1)
    # nn.init.normal_(self.W.weight,0,0.1)

  def forward(self, x):
    x = x.unsqueeze(2)#.unsqueeze(4)

    u_hat = torch.matmul(x,self.W)
    u_hat = u_hat.view(u_hat.size(0),self.num_routes, self.nc, self.out_dim)

    # c = F.softmax(self.b)
    c = F.softmax(self.b,dim=-1)
    s = (c.unsqueeze(2) * u_hat).sum(dim=1)
    v = squash(s)

    if self.r_it > 0:
      bBatch = self.b.expand((u_hat.shape[0],self.num_routes,self.nc))
      for r in range(self.r_it):
        v = v.unsqueeze(1)
        bBatch = bBatch + (u_hat * v).sum(-1)

        # c = F.softmax(bBatch.view(-1,self.nc)).view(-1,self.num_routes,self.nc,1)
        c=F.softmax(bBatch.view(-1,self.nc),dim=-1).view(-1,self.num_routes,self.nc,1)
        s = (c * u_hat).sum(dim=1)
        v = squash(s)
      
    return v, c.squeeze()

class PrimaryCaps(nn.Module):
  def __init__(self, inputChannels, outputCaps, outputDim, kernelSize, stride):
    super().__init__()
    self.conv = nn.Conv2d(inputChannels, outputCaps * outputDim, kernel_size=kernelSize, stride = stride)
    self.inputChannels = inputChannels
    self.outputCaps = outputCaps
    self.outputDim = outputDim
    self.reset_params()

  def reset_params(self):
    nn.init.normal_(self.conv.weight,0,0.1)
    # nn.init.normal(self.conv.weight,0,0.1)

  def forward(self, input):
    out = self.conv(input)
    N,C,H,W = out.size()
    out = out.view(N,self.outputCaps, self.outputDim, H, W)

    #N x OUTCAPS x OUTDIM
    out = out.permute(0,1,3,4,2).contiguous()
    out = out.view(out.size(0),-1,out.size(4))
    out = squash(out)

    return out

class ConvCaps2D(nn.Module):
  def __init__(self, nc_i, dim_i, nc_j, dim_j, kernel_size=3, stride=1,padding=1, r_num=1):
    '''
    2D Convolutional Capsule Layer. 
    Conv2DCaps is similar to a convolutional layer,
    except that its outputs will be squashed 4D tensors.
    i --> current layer.
    j --> next layer.
    Arguments
    ---
    `nc_i`: number of capsules in layer i.
    `dim_i`: dimensions of capsules in layer i.
    `nc_j`: number of capsules in layer j.
    `dim_j`: dimensions of capsules in layer j.
    `kernel_size`: Convolutional filter size.
    `stride`: convolution stride.
    `padding`: convolution padding.
    `r_num`: number of routing iterations.
    '''
    super().__init__()
    self.nc_i = nc_i
    self.dim_i = dim_i
    self.nc_j = nc_j
    self.dim_j = dim_j
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding=padding
    self.r_num=r_num

    in_channels = self.nc_i * self.dim_i
    out_channels = self.nc_j * self.dim_j

    self.conv = nn.Conv2d(in_channels,out_channels,
                          self.kernel_size,self.stride, self.padding)

  def forward(self, x):
    # x.shape: NxNCxDxHxW
    n, nc, dim, h, w = x.shape
    # Reshape x from NxNCxDxHxW --> Nx[NCxD]XHXW
    x = x.view(n,nc*dim, h, w)

    x = self.conv(x)
    h_j, w_j = x.shape[-2:]

    #reshape back to NxNCxDxHxW
    x = x.view(n,self.nc_j,self.dim_j,h_j,w_j)
    
    # Squash and return x.
    return squash(x)


class ConvCaps3D(nn.Module):
  def __init__(self, nc_i, dim_i, nc_j, dim_j,r_num=3, kernel_size=3, padding = (0,1,1)):
    '''
    3D Convolutional Capsule Layer.
    ConvCaps3D uses 3D convolutions with Dynamic Routing
    when num_routings is set greater than 1.
    i --> current layer.
    j --> next layer.
    Arguments
    ---
    `nc_i`: number of capsules in layer i.
    `dim_i`: dimensions of capsules in layer i.
    `nc_j`: number of capsules in layer j.
    `dim_j`: dimensions of capsules in layer j.
    `kernel_size`: Convolutional filter size.
    `stride`: convolution stride.
    `padding`: convolution padding.
    `r_num`: number of routing iterations.
    '''
    super().__init__()
    self.nc_i = nc_i
    self.dim_i = dim_i
    self.nc_j = nc_j
    self.dim_j = dim_j
    self.kernel_size = kernel_size
    self.r_num=r_num


    self.stride = (dim_i,1,1)
    self.padding= padding

    in_channels = 1
    out_channels = self.nc_j * self.dim_j

    self.conv3d = nn.Conv3d(in_channels,out_channels,
                            self.kernel_size,self.stride,self.padding)
    
  def forward(self, x):
    # x.shape = NxNCxDxHxW
    n, nc, dim, h, w = x.shape

    x = x.view(n,nc*dim, h, w)

    x = x.unsqueeze(1)
    x = self.conv3d(x)

    h_j, w_j = x.shape[-2:]

    x = x.view(n, self.nc_i, self.nc_j, self.dim_j ,h_j, w_j)

    # Transpose to NxHxWxDjxNCjxNCi for routing updates.

    x = x.permute(0,4,5,3,2,1)

    # B matrix for routing coefficients.
    # B.shape: NxHxWx1xNCjxNCi
    self.B = x.new(x.shape[0],h_j,w_j,1,self.nc_j,self.nc_i).to(device)

    x = self.update_routing(x, self.r_num)
    
    return x
  
  def update_routing(self, x, num_r=3):
    #x.shape = NxHxWxDjxNCjxNCi
    for ix in range(num_r):
      k = softmax3D(self.B,(1,2,3))
      s = (k * x).sum(dim=-1,keepdim=True)
      s_hat  = squash(s)

      if ix < num_r-1:
        agreements = (s_hat * x).sum(dim=3, keepdim=True)
        self.B = self.B = agreements

    s_hat = s_hat.squeeze(-1)
    batch, h_j, w_j, d_j, n_j  = s_hat.shape

    return s_hat.reshape(batch,n_j,d_j,h_j,w_j)


class MaskCID(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x, target=None):
    if target is None:
      #Inference mode
      classes = torch.norm(x,dim=2)
      pred_class = classes.max(dim=1)[1].squeeze()
    else:
      pred_class = target.max(dim=1)[1]
    
    increasing = torch.arange(start=0, end = x.shape[0]).to(device)

    m = torch.stack([increasing,pred_class], dim=1)

    masked = torch.zeros((x.shape[0],1)+x.shape[2:])
    # import pdb; pdb.set_trace()
    for i in increasing:
      masked[i] = x[m[i][0],m[i][1],:].unsqueeze(0)

    return masked.squeeze(-1), pred_class