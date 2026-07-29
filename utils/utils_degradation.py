import numpy as np
import cv2
import random
import scipy.stats as ss
from scipy import ndimage
from utils import utils_image
from skimage import transform
import cv2
import math

def noise_generator(seed=0):
    rng = np.random.default_rng(seed=seed)
    return rng

seed = 1


def add_Gaussian_noise(img, noise_level, is_clip=False):
    rng = noise_generator(seed)
    img = img.copy()
    img += rng.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    if is_clip:
        img = np.clip(img, 0.0, 1.0)
    return img



def add_Speckle_noise(img, noise_level):
    rng = noise_generator(seed)
    img = img.copy()
    img = img + img * rng.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img



def add_Re_Speckle_noise(img, noise_level):
    rng = noise_generator(seed)
    img = img.copy()
    img = img + np.sqrt(img*(noise_level / 255.0)**2 )* rng.normal(0, 1, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Re_SpeckleL_noise(img, noise_level):
    rng = noise_generator(seed)
    img = img.copy()
    img = img + np.sqrt(img*(noise_level / 255.0)**2 ) * rng.laplace(0, 1, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def generate_poisson_noise(img, scale=1.0, gray_noise=False):
    """Generate poisson noise.
    Ref: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    rng = noise_generator(seed)
    if gray_noise:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # round and clip image for counting vals correctly
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(rng.poisson(img * vals) / float(vals))
    noise = out - img
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale


def add_Poisson_noise(img, scale=1.0, clip=True, rounds=False, gray_noise=False):
    """Add poisson noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    img = img.copy()
    noise = generate_poisson_noise(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def add_mix_noise(img, noise_level=25,scale=1.0, clip=True, rounds=False, gray_noise=False):
    rng = noise_generator(seed)
    img = img.copy()
    noise1 = generate_poisson_noise(img, scale, gray_noise)
    noise2 = np.sqrt(img*(noise_level / 255.0)**2 )* rng.normal(0, 1, img.shape).astype(np.float32)
    out = img + noise1 + noise2
    
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def peaks(n):
    '''
    Implementation the peak function of matlab.
    '''
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ




def generate_gauss_kernel_mix(H, W):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    rng = noise_generator(seed)
    pch_size = 32
    K_H = math.floor(H / pch_size)
    K_W = math.floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))

    centerW = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)

    centerH = rng.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)

    scale = rng.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    out = ZZ.sum(axis=2, keepdims=False) / K
    return out

def sincos_kernel():
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(1, 20, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz


def add_sincos_kernel_noise(img, sigma_min, sigma_max):
    rng = noise_generator(seed)
    img = img.copy()
    h, w, c = img.shape
    sigma_min, sigma_max = sigma_min/255, sigma_max/255
    sigma_base = sincos_kernel()
    sigma_base = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
    sigma = cv2.resize(sigma_base, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32) # H x W
    noise = rng.normal(0, 1, img.shape) * sigma[:, :, np.newaxis]
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img



def add_peaks_noise(img, sigma_min, sigma_max):
    rng = noise_generator(seed)
    img = img.copy()
    h, w, c = img.shape
    sigma_min, sigma_max = sigma_min/255, sigma_max/255
    sigma_base = peaks(256)
    sigma_base = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
    sigma = cv2.resize(sigma_base, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32) # H x W
    noise = rng.normal(0, 1, img.shape) * sigma[:, :, np.newaxis]
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img


def add_gauss_kernel_noise(img, sigma_min, sigma_max):
    rng = noise_generator(seed)
    img = img.copy()
    h, w, c = img.shape
    sigma_min, sigma_max = sigma_min/255, sigma_max/255
    sigma_base = generate_gauss_kernel_mix(256, 256)
    sigma_base = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
    sigma = cv2.resize(sigma_base, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32) # H x W
    noise = rng.normal(0, 1, img.shape) * sigma[:, :, np.newaxis]
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img




#########################################3#########################################3
def re_sincos_kernel(h, w):
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, w), np.linspace(1, 20, h))
    zz = np.sin(xx) + np.cos(yy)
    return zz

def re_peaks(h, w):
    '''
    Implementation the peak function of matlab.
    '''
    X = np.linspace(-2, 2, w)
    Y = np.linspace(-2, 2, h)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ


def re_gauss_kernel_mix(H, W, num_kernels=5):#5
    """
    Generate a H x W Gaussian mixture kernel.
    Each component is defined by its mean (center) and std (scale).

    Args:
        H: int - Height of the output kernel
        W: int - Width of the output kernel
        num_kernels: int - Number of Gaussian kernels in the mixture
        seed: int - Random seed for reproducibility (optional)

    Returns:
        out: numpy.ndarray - The mixed Gaussian kernel of shape (H, W)
    """

    rng = np.random.default_rng(seed)
    # rng = np.random.default_rng(10)

    # Randomly sample means (centers) within the image size
    centers_x = rng.uniform(low=0, high=W, size=(num_kernels,))
    centers_y = rng.uniform(low=0, high=H, size=(num_kernels,))

    # Randomly sample standard deviation (scale) for each kernel
    scales = rng.uniform(low=min(H, W) * 0.2, high=min(H, W) * 0.5, size=(num_kernels,))

    # Create meshgrid for the coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Initialize the mixed Gaussian kernel
    kernel_mix = np.zeros((H, W), dtype=np.float32)

    for i in range(num_kernels):
        # Compute Gaussian kernel for each component
        scale = scales[i]
        center_x = centers_x[i]
        center_y = centers_y[i]

        # Gaussian distribution formula
        gauss_kernel = (1. / (2 * np.pi * scale**2)) * np.exp(
            -((X - center_x)**2 + (Y - center_y)**2) / (2 * scale**2)
        )

        # Accumulate kernels
        kernel_mix += gauss_kernel

    # Normalize by the number of kernels to ensure consistent scale
    kernel_mix /= num_kernels

    return kernel_mix


def add_Speckle_func_noise(img, sigma_min, sigma_max, func):
    rng = noise_generator(seed)
    img = img.copy()
    h, w, c = img.shape
    sigma_min, sigma_max = sigma_min/255, sigma_max/255
    if func == 'sincos':
        sigma_base = re_sincos_kernel(h, w)
    if func == 'peaks':
        sigma_base = re_peaks(h, w)
    if func == 'mixgauss':
        sigma_base = re_gauss_kernel_mix(h, w)
    
    sigma = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
    sigma = sigma[:, :, np.newaxis]
    
    noise = rng.normal(0, 1, img.shape) * np.sqrt(img*sigma**2)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img



def add_SpeckleL_func_noise(img, sigma_min, sigma_max, func):
    rng = noise_generator(seed)
    img = img.copy()
    h, w, c = img.shape
    sigma_min, sigma_max = sigma_min/255, sigma_max/255
    if func == 'sincos':
        sigma_base = re_sincos_kernel(h, w)
    if func == 'peaks':
        sigma_base = re_peaks(h, w)
    if func == 'mixgauss':
        sigma_base = re_gauss_kernel_mix(h, w)
    
    sigma = sigma_min + (sigma_base-sigma_base.min())/(sigma_base.max()-sigma_base.min()) * (sigma_max-sigma_min)
    sigma = sigma[:, :, np.newaxis]
    
    noise = rng.laplace(0, 1, img.shape) * np.sqrt(img*sigma**2)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img



