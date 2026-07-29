import torch
import torch.nn as nn

import numpy as np
import random
from torch.nn import functional as F
from einops import rearrange



class ResBlock(nn.Module):
    def __init__(self, dim, num_heads,bias=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias),
        )

    def forward(self, x):
        return x + self.layers(x)

    


class Net(nn.Module):

    def __init__(self, img_channel=3, width=64, middle_blk_num=4, refine_blk_num=6,enc_blk_nums=[4,4,4], dec_blk_nums=[4,4,4],norm=True):
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
                    *[ResBlock(chan, 2 * (chan // width)) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2, bias=False)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[ResBlock(chan, 2 * (chan // width)) for _ in range(middle_blk_num)]
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
                    *[ResBlock(chan, 2 * (chan // width)) for _ in range(num)]
                )
            )
        
        self.refine =  nn.Sequential(
                    *[ResBlock(chan, 2 * (chan // width)) for _ in range(refine_blk_num)]
                )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        b, c, h, w = inp.shape
        inp = self.check_image_size(inp)
        
        if self.pre_mask:
            if prob is None:
                prob = random.uniform(self.mask_ratio[0], self.mask_ratio[1]) 
            mask = (torch.rand(b, 1, h, w) > prob).float().to(inp.device)
            inp = inp * mask
        
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
    model = Net().cuda(1)
    macs, params = get_model_complexity_info(model, (3,256,256), print_per_layer_stat=False)
    print(macs)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))