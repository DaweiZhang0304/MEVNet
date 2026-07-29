import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange
from utils.utils_loss import TVLoss,   Uncertainty_L1, WassLoss,WassLoss
import random
import importlib
from utils.utils_image import mixup_data
import numpy as np

class Solver():
    def __init__(self, module, opt):
        self.opt = opt
        self.gpu_id = opt['gpu_id']
        self.net_G = module.Net(**opt['network_g']).cuda(self.gpu_id[0])
        torch.cuda.set_device(self.gpu_id[0])
        self.net_G = nn.DataParallel(self.net_G, device_ids=self.gpu_id, output_device=self.gpu_id[0])
        model_name = opt['model_name']
        self.writer = SummaryWriter(log_dir=f'logs/{model_name}_train')
        os.makedirs(opt['path'].get('model_save_path'),exist_ok=True)

        
        if 'isp' in opt['dataset'].get('noise_type'):
            from utils.utils_data import TrainDatasetISP as TrainDataset
        if 'gauss' in opt['dataset'].get('noise_type'):
            from utils.utils_data import TrainDataset as TrainDataset

        print("# params:", sum(map(lambda x: x.numel(), filter(lambda p: p.requires_grad, self.net_G.parameters()))))
        #######loss##########
        self.L1_loss = nn.L1Loss()
        self.wass_loss = WassLoss(opt['train'].get('spatial_freq_weight'))
        #######optim and scheduler##########
        self.optim_G = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.net_G.parameters()), opt['train'].get('lr'),
                betas=(0.9, 0.999), eps=1e-8
            )
        
        if opt['train'].get('use_scheduler'):
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optim_G, 
                                                             opt['train'].get('decay_step'), 
                                                             gamma=opt['train'].get('gamma'))
        #######checkpoint resume##########
        self.start_step = 0
        if opt['path'].get('ckpt_resume_path') is not None:
            ckpt = torch.load(opt['path'].get('ckpt_resume_path'))
            self.net_G.load_state_dict(ckpt['model_state_dict'])
            self.optim_G.load_state_dict(ckpt['optimizer_state_dict'])
            self.start_step = ckpt['step']
            self.scheduler_G.last_epoch = self.start_step
            print(f'scheduler.last_step={self.scheduler_G.last_epoch}')


        self.train_loader = torch.utils.data.DataLoader(
            TrainDataset(opt['dataset']),
            batch_size = opt['train'].get('batch_size'),
            num_workers = opt['train'].get('num_workers'),
            shuffle = False,
            drop_last = True,
             pin_memory=True,
        )

    def fit(self):
        opt = self.opt
        train_param = opt['train']
        max_steps = train_param.get('max_steps')
        t = tqdm(range(self.start_step, max_steps),ncols=100) 
        for step in t:
            
            record_loss_G = 0
            record_loss_D = 0
            for i, inputs in enumerate(self.train_loader):
                self.optim_G.zero_grad()   
                HQ = inputs[0].cuda(self.gpu_id[0])
                LQ = inputs[1].cuda(self.gpu_id[0])
                
                if 'ntnet' in opt['model_name']:
                    sigma_map = inputs[2].cuda(self.gpu_id[0])
                    k = np.random.choice([2,4,8], size=1)[0]
                    trans_LQ = self.net_G(LQ, k)
                    loss_total_G = self.wass_loss(trans_LQ, HQ, sigma_map)
                else:
                    gen_HQ = self.net_G(LQ)
                    loss_total_G = self.L1_loss(gen_HQ , HQ)
                
                loss_total_G.backward()
                self.optim_G.step()

                record_loss_G += loss_total_G
                 
            if train_param.get('use_scheduler'):
                self.scheduler_G.step()

            self.writer.add_scalar(tag="loss_total",
                    scalar_value=record_loss_G.detach()/(i+1),  # 纵坐标的值
                    global_step=step  
                    )
            

            self.summary(record_loss_G/(i+1), record_loss_D/(i+1),t)
            if (step+1) % opt['train'].get('ckpt_steps') == 0:
                self.save_ckpt(self.opt['model_name'],
                               self.opt['path'].get('ckpt_path'),
                               step)

        self.save_ckpt(self.opt['model_name'],
                       self.opt['path'].get('model_save_path'))
        self.writer.close()

    def summary(self, loss_total_G, loss_total_D, t):
        curr_lr = self.optim_G.param_groups[0]['lr']
        t.set_postfix_str('(loss_total_G:{:.5f},loss_total_D:{:.5f},LR: {}'.
                          format(loss_total_G,loss_total_D,curr_lr))

    def save_ckpt(self, model_name,save_path,step=None):
        if step is not None:
            os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
            checkpoint = {'model_state_dict': self.net_G.state_dict(),
                          'optimizer_state_dict': self.optim_G.state_dict(),
                          'step': step}
            save_path = os.path.join(save_path, model_name, f'{model_name}_{step + 1}.pth')
        else:
            checkpoint = {'model_state_dict': self.net_G.state_dict()}
            save_path = os.path.join(save_path, f'{model_name}.pth')
        torch.save(checkpoint, save_path)

    def add_noise_to_weights(self, model, noise_std):
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                noise = torch.randn_like(module.weight) * noise_std
                module.weight.data += noise





def transfer_model(pretrained_file, model, gpu_id, frozen=True):
    pretrained_dict = torch.load(pretrained_file, 
                                 map_location=lambda storage, loc: storage.cuda(gpu_id))['model_state_dict']  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    if frozen:
        model = frozen_state_dict(model)
    return model

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict


def frozen_state_dict(model):
    for k, v in model.named_parameters():
        if 'post_refine' not in k:
            v.requires_grad = False
    return model