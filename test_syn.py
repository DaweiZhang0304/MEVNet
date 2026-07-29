import importlib
import os
import torch
import torch.nn.functional as F
from torch import nn
from skimage import io, util
from tqdm import tqdm
import yaml
from utils import utils_metric, utils_image, utils_degradation
import numpy as np
import warnings
from natsort import natsorted
import logging
from PIL import Image
warnings.filterwarnings('ignore')

def cal_metric(out, HQ):
    psnr = utils_metric.calculate_psnr(out, HQ)
    ssim = utils_metric.calculate_ssim(out, HQ)
    return psnr, ssim

def save_img(img, img_name, save_root='./results/'):
    save_path = os.path.join(save_root, img_name)
    io.imsave(save_path, img)
    

@torch.no_grad()
def main():
    with open('option/test/mevnet.yml', 'r', encoding='utf-8') as f:  # 读yaml文件，编码用utf-8
        cfg = f.read()  # 读全部文件
        opt = yaml.load(cfg, Loader=yaml.FullLoader)

    set_id = 2
    postfix = ''
    current_epoch = ''  
    model_name = opt['model_name'].lower()
    paths_HQ = opt['dataset'][f'test_{set_id}']['gt_path']
    dataset_name =  opt['dataset'][f'test_{set_id}']['name']
    image_names = natsorted(os.listdir(paths_HQ))
    if current_epoch == '':
        pretrain = os.path.join(opt['path']['model_save_path'],f'{model_name}{postfix}.pth')
    else:
        pretrain = os.path.join(opt['path']['ckpt_path'],f'{model_name}{postfix}',f'{model_name}{postfix}{current_epoch}.pth')
    gpu_id = opt['gpu_id']

    module = importlib.import_module("model.{}".format(model_name))
    net = module.Net(**opt['network_g']).cuda(gpu_id[0])
    net = nn.DataParallel(net, device_ids=gpu_id, output_device=gpu_id[0])
    best_ckpt = torch.load(pretrain, map_location=lambda storage, loc: storage.cuda(gpu_id[0]))
    net.load_state_dict(best_ckpt["model_state_dict"])
    net.eval()
    
    noise_dic = {
        'Gaussian':[20],
        'Re_Speckle':[60, 80, 90],
        'Poisson': [4, 5, 6],
        'mix':[[90, 6]],
         'Speckle_sincos':[[60, 120]],
          'Speckle_peaks':[[60, 120]],
          'Speckle_mixguass':[[60, 120]],
    }

    os.makedirs('./metric_results',exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename = f'metric_results/test_{model_name}{postfix}.log',
                    )  
    logger = logging.getLogger('test')
    
    
    N = len(image_names)
    for k, v in noise_dic.items():
        for level in v:
            save_path = os.path.join('results',dataset_name,f'{model_name}{postfix}{current_epoch}',k, str(level))
            os.makedirs(save_path,exist_ok=True)
            print(save_path)
            logger.info(save_path)
    
            psnr_total, ssim_total= 0, 0
            for i,name in enumerate(image_names):
                HQ = np.array(Image.open(os.path.join(paths_HQ, name)))
                HQ = utils_image.uint2single(HQ)
                
                if k == 'Speckle':
                    LQ = utils_degradation.add_Speckle_noise(HQ, level)
                if k == 'Re_Speckle':
                    LQ = utils_degradation.add_Re_Speckle_noise(HQ, level)
                if k == 'Gaussian':
                    LQ = utils_degradation.add_Gaussian_noise(HQ, level)
                if k == 'Poisson':
                    LQ = utils_degradation.add_Poisson_noise(HQ, level)
                if k == 'sincos':
                    LQ = utils_degradation.add_sincos_kernel_noise(HQ, level[0],level[1])
                if  k == 'mix':
                    LQ = utils_degradation.add_mix_noise(HQ, level[0],level[1])
                if 'Speckle_' in k:
                    LQ = utils_degradation.add_Speckle_func_noise(HQ, level[0],level[1], k.split('_')[-1])

                LQ = utils_image.im2tensor(LQ).unsqueeze(0).cuda(gpu_id[0])
                HQ = utils_image.im2tensor(HQ).unsqueeze(0).cuda(gpu_id[0])
                out =net(LQ)
                LQ = utils_image.tensor2im(LQ.squeeze(0).cpu())
                out = utils_image.tensor2im(out.squeeze(0).cpu())
                HQ = utils_image.tensor2im(HQ.squeeze(0).cpu())
                current_psnr, current_ssim = cal_metric(out, HQ)
                psnr_total += current_psnr
                ssim_total += current_ssim
                msg1 = f'idx={i}, {name}===>psnr:{current_psnr:.2f}, ssim:{current_ssim:.4f}'
                logger.info(msg1)
                save_img(out, name, save_path)
                save_img(LQ,'noisy_'+name, save_path)
                
            msg_final = f'psnr_avg/ssim_avg = &{psnr_total/N:.2f}/{ssim_total/N:.4f} \n #########'
            logger.info(msg_final)
            print(msg_final)




if __name__ == "__main__":
    main()
