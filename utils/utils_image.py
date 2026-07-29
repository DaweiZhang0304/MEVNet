import random
import numpy as np
import math
import cv2
import torch
from torch.nn import functional as F
from einops import rearrange

def random_crop(HQ, psize):
    h, w = HQ.shape[:-1]
    x = random.randrange(0, w-psize+1)#x,y 是patch左上角的坐标
    y = random.randrange(0, h-psize+1)
    crop_HQ = HQ[y:y+psize, x:x+psize]
    return crop_HQ.copy()


def center_crop(HQ, psize):
    h, w = HQ.shape[:-1]
    start = h//2 - (psize//2)
    center_crop = HQ[start:start+psize, start:start+psize]
    return center_crop.copy()
    
    

def flip_and_rotate(HQ):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hflip:
        HQ= HQ[:, ::-1, :]
    if vflip:
        HQ= HQ[::-1, :, :]
    if rot90:
        HQ= HQ.transpose(1, 0, 2)

    return HQ




def random_crop_pair(HQ, LQ, psize):
    h, w = HQ.shape[:-1]
    x = random.randrange(0, w-psize+1)#x,y 是patch左上角的坐标
    y = random.randrange(0, h-psize+1)
    crop_HQ = HQ[y:y+psize, x:x+psize]
    crop_LQ = LQ[y:y+psize, x:x+psize]
    return crop_HQ.copy(), crop_LQ.copy()

    
def flip_and_rotate_pair(HQ, LQ):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    if hflip:
        HQ = HQ[:, ::-1, :]
        LQ = LQ[:, ::-1, :]
    if vflip:
        HQ= HQ[::-1, :, :]
        LQ= LQ[::-1, :, :]
    if rot90:
        HQ= HQ.transpose(1, 0, 2)
        LQ= LQ.transpose(1, 0, 2)
    return HQ, LQ

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t.astype(np.float32))
    return tensor

def tensor2im(tensor):
    return single2uint(tensor.permute(1, 2, 0).numpy())





##################################################################################


def pixel_shuffle_down_sampling_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # unshuffled = F.pixel_unshuffle(x, f)
        # if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), 'reflect')
        unshuffled = unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 2, 3, 1, 4, 5).contiguous()
        unshuffled = unshuffled.view(-1, c, w // f + 2 * pad, h // f + 2 * pad).contiguous()
        return unshuffled
        


def pixel_shuffle_up_sampling_pd(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        # if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        b = b // (f * f)
        before_shuffle = x.view(b, f, f, c, w, h)
        before_shuffle = before_shuffle.permute(0, 3, 1, 2, 4, 5).contiguous()
        before_shuffle = before_shuffle.view(b, c*f*f, w, h)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    
    
    
##################################################################################



def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

##################################################################################


operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)
    
def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage






# def generate_indices(shape):
#     indices = torch.stack([torch.randperm(shape[-1]) for _ in range(shape[-2])])
#     expanded_indices = indices.view(1, 1, shape[-2], shape[-1]).expand(*shape[:-1], shape[-1])
#     return expanded_indices


# def Random_PD(x, pd, indices=None,random=True):
#     b, c, h, w =x.shape
#     x = rearrange(x, 'b c (h k1) (w k2) -> b c (h w) (k1 k2)',h=h//pd,w=w//pd,k1=pd,k2=pd)
#     if random:
#         if indices is None:
#             indices = generate_indices(x.size()).to(x.device) 
#         x = x.gather(-1, indices)
#         x = rearrange(x, 'b c (h w) (k1 k2) -> (k1 k2) b c h w', h=h//pd, w=w//pd,k1=pd,k2=pd)
#         return x, indices
#     else:
#         x = rearrange(x, 'b c (h w) (k1 k2) -> (k1 k2) b c h w', h=h//pd, w=w//pd,k1=pd,k2=pd)
#         return x





def pixel_unshuffle(input, factor):
    """
    (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)

    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

def pixel_shuffle(input, factor):
    """
    (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    """
    batch_size, channels, in_height, in_width = input.size()


    out_height = in_height * factor
    out_width = in_width * factor

    batch_size /= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)

    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)




def mixup_data(x, y, alpha=0.4, use_cuda=True):
    dist = torch.distributions.beta.Beta(torch.tensor([0.4]), torch.tensor([0.4]))
    lam = dist.rsample((1, 1)).item()
    r_index = torch.randperm(y.size(0))
    mixed_y = lam * y + (1 - lam) * y[r_index, :]
    mixed_x = lam * x + (1 - lam) * x[r_index, :]
    return torch.clip(mixed_x, 0., 1.), torch.clip(mixed_y, 0., 1.)