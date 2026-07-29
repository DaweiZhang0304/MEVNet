import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim=32, output_dim=32, head_dim=16, window_size=8, type='W'):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=False)


        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=False)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        orign_H, orign_W = x.shape[-2:]
        x= self.check_image_size(x)
        x = rearrange(x, 'b c h w -> b h w c')
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = q @ k.transpose(-2, -1) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        output = rearrange(output, 'b h w c -> b c h w')
        return output[:,:,:orign_H, :orign_W]

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x








class ConstantNorm(nn.Module):
    def __init__(self, dim):
        super().__init__() 
        self.scale = dim ** -0.5
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        
    def forward(self, x):
        return x * self.scale * self.weight



class HalfMod(nn.Module):
    def __init__(self, dim, head_dim, window_size, type_, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.cnorm = ConstantNorm(dim//2)
        self.affine = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False, groups=2)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1,bias=False)
        self.attn = WMSA(dim//2, dim//2, head_dim, window_size, type_)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.cnorm(x2)
        gamma, beta = self.affine(x1).chunk(2, dim=1)
        
        x1 = self.attn(x1)
        u = x1.mean(1, keepdim=True)
        s = (x1 - u).pow(2).mean(1, keepdim=True)
        x1 = (x1 - u) / torch.sqrt(s + self.eps)
        
        x1 = torch.mul(gamma, x1) + beta
        x = self.conv(torch.cat([x1, x2],dim=1))
        return x 
  
  
      
        
class ResBlock(nn.Module):
    def __init__(self, dim, head_dim, window_size=8, type_='W', bias=False):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1,bias=bias)
        self.conv2 = nn.Conv2d(dim //2, dim  //2, kernel_size=3, padding=1,bias=bias)
        self.conv3 = nn.Conv2d(dim // 2, dim, kernel_size=1,bias=bias)
        self.hm = HalfMod(dim, head_dim, window_size, type_)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.hm(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv2(x1)
        s = torch.sqrt(torch.var(x1, dim=1, keepdim=True)+torch.var(x2, dim=1, keepdim=True)+1e-6) 
        x = torch.mul(x1, x2) / s
        x = self.conv3(x)
        return x + shortcut


class Net(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=4, refine_blk_num=6,
                 enc_blk_nums=[4,4,4], dec_blk_nums=[4,4,4], norm=True):
        super().__init__()
        
        self.norm = norm
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,bias=False)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,bias=False)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[ResBlock(chan, width//2, type_='W' if not i%2 else 'SW') for i in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2, bias=False)
            )
            chan = chan * 2

        
        self.middle_blks = \
            nn.Sequential(
                *[ResBlock(chan, width//2, type_='W' if not i%2 else 'SW') for i in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan *2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                            )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[ResBlock(chan, width//2, type_='W' if not i%2 else 'SW') for i in range(num)]
                )
            )
        
        self.refine =  nn.Sequential(
                    *[ResBlock(chan, width//2, type_='W' if not i%2 else 'SW') for i in range(refine_blk_num)]
                )

        self.padder_size = 2 ** len(self.encoders)


    def forward(self, inp, mask=None):
        b, c, h, w = inp.shape
        
        inp = self.check_image_size(inp)
        if self.norm:
            mean = torch.mean(inp, dim=(-1,-2),keepdim=True)
            max_ = torch.max(inp.flatten(2), dim=-1, keepdim=True)[0].unsqueeze(-1)
            min_ = torch.min(inp.flatten(2), dim=-1, keepdim=True)[0].unsqueeze(-1)
            range_ = max_ - min_
            inp = (inp - mean)/range_
        else:
            mean = torch.mean(inp, dim=(-1,-2),keepdim=True)
            inp = inp - mean
            

        x = self.intro(inp)
        if mask is not None:
            x = mask * x
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.refine(x)
        x = self.ending(x)
        x = x + inp
        
            
        if self.norm:
            x = range_ * x + mean
        else:
            x = x + mean

        return x[:, :, :h, :w]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x

    

if __name__  == '__main__':
    # torch.cuda.empty_cache()
    from ptflops import get_model_complexity_info
    model = Net().cuda(7)
    macs, params = get_model_complexity_info(model, (3,256,256), print_per_layer_stat=False)
    print(macs)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))