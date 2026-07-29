import importlib
import torch
import yaml
import argparse
from solver import Solver
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option",type=str)
    parser.add_argument("--postfix",type=str, default='')
    args = parser.parse_args()

    #'./option/baseline.yml'
    with open(args.option, 'r', encoding='utf-8') as f:  # 读yaml文件，编码用utf-8
        cfg = f.read()  # 读全部文件
        opt = yaml.load(cfg, Loader=yaml.FullLoader)

    torch.manual_seed(opt['seed'])
    #获取model文件夹下的模型
    module = importlib.import_module("model.{}".format(opt['model_name'].lower()))
    opt['model_name'] = opt['model_name'] + args.postfix
    print('final model name: ', opt['model_name'])
    solver = Solver(module, opt)
    solver.fit()

if __name__ == "__main__":
    main()
