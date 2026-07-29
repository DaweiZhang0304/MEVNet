import importlib
import os
import torch
from torch import nn
from skimage import io, util
import yaml
from utils import utils_metric, utils_image
import warnings
from natsort import natsorted
import logging
warnings.filterwarnings('ignore')


def cal_metric(out, HQ):
    psnr = utils_metric.calculate_psnr(out, HQ)
    ssim = utils_metric.calculate_ssim(out, HQ)
    return psnr, ssim

def save_img(img, img_name, save_root='./results/'):
    if '_mean' in img_name:img_name = img_name.replace('_mean','')
    save_path = os.path.join(save_root, img_name)
    io.imsave(save_path, img)



@torch.no_grad()
def main(model_type):
        
    with open(f'option/test/ntnet.yml', 'r', encoding='utf-8') as f:  # 读yaml文件，编码用utf-8
        cfg = f.read()  # 读全部文件
        opt_nt = yaml.load(cfg, Loader=yaml.FullLoader)

    postfix = ''
    current_epoch = f'' 
    model_name = opt_nt['model_name'].lower()
    if current_epoch == '':
        pretrain = os.path.join(opt_nt['path']['model_save_path'],f'{model_name}{postfix}.pth')
    else:
        pretrain = os.path.join(opt_nt['path']['ckpt_path'],f'{model_name}{postfix}',f'{model_name}{postfix}{current_epoch}.pth')

    gpu_id = opt_nt['gpu_id']  # 测试使用 GPU 7

    torch.cuda.set_device(gpu_id[0])
    model_name = opt_nt['model_name'].lower()  
    module = importlib.import_module("model.{}".format(model_name))
    ntnet = module.Net(**opt_nt['network_g']).cuda(gpu_id[0])
    ntnet = nn.DataParallel(ntnet, device_ids=gpu_id, output_device=gpu_id[0])
    best_ckpt = torch.load(pretrain, map_location=lambda storage, loc: storage.cuda(gpu_id[0]))
    ntnet.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    ntnet.eval()  
    

    with open(f'option/train/{model_type}.yml', 'r', encoding='utf-8') as f:  # 读yaml文件，编码用utf-8
        cfg = f.read()  # 读全部文件
        opt_dn = yaml.load(cfg, Loader=yaml.FullLoader)

    denoiser_name = opt_dn['model_name']
    module = importlib.import_module("model.{}".format(opt_dn['model_name'].lower()))
    denoiser = module.Net(**opt_dn['network_g']).cuda(gpu_id[0])
    denoiser = nn.DataParallel(denoiser, device_ids=gpu_id, output_device=gpu_id[0])

    denoiser.load_state_dict(
            torch.load(os.path.join('./experiments',opt_dn['model_name'].lower()+'.pth'), 
                    map_location=lambda storage, loc: storage.cuda(gpu_id[0]))["model_state_dict"]
    )
    set_id = 1
    benchmark =opt_nt['dataset'][f'test_{set_id}']['name']
    HQ_path = opt_nt['dataset'][f'test_{set_id}']['HQ_path']
    LQ_path = opt_nt['dataset'][f'test_{set_id}']['LQ_path']

    HQ_name = natsorted(os.listdir(HQ_path))
    LQ_name = natsorted(os.listdir(LQ_path))
    
    save_path = os.path.join(f'results_{model_name}_{benchmark}', f'{denoiser_name}_'+model_name + postfix, 'epoch'+current_epoch)
    os.makedirs(save_path,exist_ok=True)
    print(save_path)
    
    os.makedirs('./metric_results',exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename = f'metric_results/test_real_{denoiser_name}_{model_name}{postfix}.log',
                    filemode='a')
    logger = logging.getLogger('test')
    logger.info(save_path)
    

    N = len(HQ_name)
    psnr_total, ssim_total = 0, 0

    for i in range(N):
        HQ = io.imread(os.path.join(HQ_path, HQ_name[i]))
        HQ = utils_image.uint2single(HQ)
        
        LQ = io.imread(os.path.join(LQ_path, LQ_name[i]))
        LQ = utils_image.uint2single(LQ)
        
        LQ = utils_image.im2tensor(LQ).unsqueeze(0).cuda(gpu_id[0])
        HQ = utils_image.im2tensor(HQ).unsqueeze(0).cuda(gpu_id[0])
        
    
        trans_LQ = ntnet(LQ, k=8)
        out = denoiser(trans_LQ)
        

        trans_LQ = utils_image.tensor2im(trans_LQ.squeeze(0).cpu())
        out = utils_image.tensor2im(out.squeeze(0).cpu())
        HQ = utils_image.tensor2im(HQ.squeeze(0).cpu())
        current_psnr, current_ssim = cal_metric(out, HQ)
        psnr_total += current_psnr
        ssim_total += current_ssim

        
        msg1 = f'{HQ_name[i]}===>psnr:{current_psnr:.2f}, ssim:{current_ssim:.4f}'
        logger.info(msg1)
        # print(msg1)
        save_img(out, HQ_name[i], save_path)
        
    msg_final = f'psnr_avg/ssim_avg = &{psnr_total/N:.2f}/{ssim_total/N:.4f}\n #########'
    logger.info(msg_final)
    print(msg_final)

if __name__ == "__main__":
    for name in [
        'mevnet',
        'mevnet_sl_elu'
    ]:
        main(name)
