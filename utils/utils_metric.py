import random
import numpy as np
import math
import cv2
# import lpips


def convert_rgb_to_y(img):#old
    return 16.0 + (65.738 * img[:,:,0] + 129.057 * img[:,:,1] + 25.046 * img[:,:,2])/256.0

def rgb2ycbcr(img, y_only=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.

    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214]]
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(im1, im2, border=0, ycbcr=False):
    '''
    SSIM the same outputs as MATLAB's
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    if im1.ndim == 2:
        return ssim(im1, im2)
    elif im1.ndim == 3:
        if im1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(im1[:,:,i], im2[:,:,i]))
            return np.array(ssims).mean()
        elif im1.shape[2] == 1:
            return ssim(np.squeeze(im1), np.squeeze(im2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    '''
    PSNR metric.
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: uint8 [0,255] or float [0,1]
        only_y: only return Y channel
    '''
    # transform to float64 data type, range [0, 255]
    if im.dtype == np.uint8:
        im_temp = im.astype(np.float64)
    else:
        im_temp = (im * 255).astype(np.float64)

    # convert
    if only_y:
        rlt = np.dot(im_temp, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im_temp, np.array([[65.481,  -37.797, 112.0  ],
                                           [128.553, -74.203, -93.786],
                                           [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if im.dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(im.dtype)


# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# def calculate_lpips(img1, img2, gpu):
#     global loss_fn_alex
#     loss_fn_alex = loss_fn_alex.cuda(gpu)
#     return loss_fn_alex(img1, img2)







