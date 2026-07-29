import os
import glob
import skimage.io as io
import skimage.color as color
import torch
from utils import utils_image
import numpy as np
import random
import torch
import cv2
from natsort import natsorted
from skimage.util import random_noise
from ISP_implement import ISP

# from ISPG_implement import ISPG

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.HQ_paths = sorted(glob.glob(os.path.join(opt.get('gt_path'), "*.png")))  # 查找所有后缀名为png的文件
        self.opt = opt
        # self.flag = 0

    def __getitem__(self, index):
        index = index % len(self.HQ_paths)
        HQ = io.imread(self.HQ_paths[index])
        if len(HQ.shape) < 3:
            HQ = color.gray2rgb(HQ)

        HQ= utils_image.random_crop(HQ, self.opt.get('crop_size'))
        HQ= utils_image.flip_and_rotate(HQ)
        HQ = utils_image.uint2single(HQ)

        low, high = self.opt.get('noise_level')
        noise_level = random.randint(low, high)
        LQ = HQ + np.random.normal(0, noise_level / 255.0, HQ.shape).astype(np.float32)

        LQ = utils_image.im2tensor(LQ)
        HQ = utils_image.im2tensor(HQ)
        return HQ, LQ

    def __len__(self):
        return len(self.HQ_paths)
    
    
class TrainDatasetISP(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.HQ_paths = sorted(glob.glob(os.path.join(opt.get('gt_path'), "*.png")))  # 查找所有后缀名为png的文件
        self.opt = opt
        # self.flag = 0
        self.isp = ISP()

    def __getitem__(self, index):
        index = index % len(self.HQ_paths)
        HQ = io.imread(self.HQ_paths[index])
        if len(HQ.shape) < 3:
            HQ = color.gray2rgb(HQ)

        HQ= utils_image.random_crop(HQ, self.opt.get('crop_size'))
        HQ= utils_image.flip_and_rotate(HQ)
        HQ = utils_image.uint2single(HQ)

        HQ, LQ, sigma_total = self.isp.noise_generate_srgb(HQ)
        return utils_image.im2tensor(HQ),utils_image.im2tensor(LQ), torch.tensor(sigma_total).unsqueeze(0).float()

    def __len__(self):
        return len(self.HQ_paths)
    
    
