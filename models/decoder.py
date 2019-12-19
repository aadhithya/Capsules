import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Interpolator(nn.Module):
    """
    Interpolate by de/up/backward convolution with a bilinear kernel.
    Take
        channel_dim: the input channel dimension
        rate: upsampling rate, that is 4 -> 4x upsampling
        odd: the kernel parity, which is too much to explain here for now, but
             will be handled automagically in the future, promise.
        normalize: whether kernel sums to 1
    """
    def __init__(self, channel_dim, rate, odd=True, normalize=False):
        super().__init__()
        self.rate = rate
        ksize = rate * 2
        if odd:
            ksize -= 1
        # set weights to within-channel bilinear interpolation
        kernel = torch.from_numpy(self.bilinear_kernel(ksize, normalize))
        weight = torch.zeros(channel_dim, channel_dim, ksize, ksize)
        for k in range(channel_dim):
            weight[k, k] = kernel
        # fix weights
        self.weight = nn.Parameter(weight, requires_grad=False)
        
        
    def bilinear_kernel(self,size, normalize=False):
        """
        Make a 2D bilinear kernel suitable for upsampling/downsampling with
        normalize=False/True. The kernel is size x size square.
        Take
            size: kernel size (square)
            normalize: whether kernel sums to 1 (True) or not
        Give
            kernel: np.array with bilinear kernel coefficient
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        kernel = (1 - abs(og[0] - center) / factor) * \
                 (1 - abs(og[1] - center) / factor)
        if normalize:
            kernel /= kernel.sum()
        return kernel

    def forward(self, x):
        # no groups (for speed with current pytorch impl.) and no bias
        return F.conv_transpose2d(x, self.weight, stride=self.rate)   
    
class ConvDecoder(nn.Module):
    """
    FCN-32s like fully convolutional decoder network
    """

    def __init__(self, in_channel=512, out_channel=2, imsize=(256,256), interp_rate=32):
        super().__init__()
        
        self.h, self.w=imsize
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel//2, in_channel//4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel//4, in_channel//8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel//8, in_channel//16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel//16, out_channel, kernel_size=6, stride=2),
        )
        
        self.interpolator = Interpolator(out_channel, interp_rate, odd=False)
        
    
    def resize(self,inp):
        
        H,W=inp.size()[-2:]
        startx = H//2 - self.h//2
        starty = W//2 - self.w//2  
        out = inp[:,:,startx:startx + h, starty:starty + w]
        
        return out
        
    def forward(self, inp):
        
        inp=self.decoder(inp)
        inp=self.interpolator(inp)
        out= self.resize(inp)

        return out

