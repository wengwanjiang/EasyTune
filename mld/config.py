import os
import importlib
from typing import Type, TypeVar
from argparse import ArgumentParser

from omegaconf import OmegaConf, DictConfig


def get_module_config(cfg_model: DictConfig, paths: list[str], cfg_root: str) -> DictConfig:
    files = [os.path.join(cfg_root, 'modules', p+'.yaml') for p in paths]
    for file in files:
        assert os.path.exists(file), f'{file} is not exists.'
        with open(file, 'r') as f:
            cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: DictConfig) -> TypeVar:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    

def parse_args() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")
    parser.add_argument('--example', type=str, required=False, help="The input texts and lengths with txt format")
    parser.add_argument('--example_hint', type=str, required=False, help="The input hint ids and lengths with txt format")
    parser.add_argument('--ckpt', type=str, required=False, help="The input hint ids and lengths with txt format")
    parser.add_argument('--no-plot', action="store_true", required=False, help="Whether to plot the skeleton-based motion")
    parser.add_argument('--replication', type=int, default=1, help="The number of replications of sampling")
    parser.add_argument('--vis', type=str, default="tb", choices=['tb', 'swanlab'], help="The visualization backends: tensorboard or swanlab")
    parser.add_argument('--optimize', action='store_true', help="Enable optimization for motion control")
    
    parser.add_argument('--spm_path', type=str, required=False, help="")
    parser.add_argument('--mld_path', type=str, required=False, help="")

    parser.add_argument('--ft_m', type=int, default=None)
    parser.add_argument('--ft_prob', type=float, default=None)
    parser.add_argument('--ft_t', type=int, default=None)
    parser.add_argument('--ft_dy', type=int, default=2)
    parser.add_argument('--ft_k', type=int, default=None)
    parser.add_argument('--ft_skip', type=str2bool, default=False)
    parser.add_argument('--ft_reverse', type=str2bool, default=False)
    parser.add_argument('--ft_custom', type=str, default=None)
    parser.add_argument('--ft_type', type=str, default=None, choices=['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'EZtune', 'None'])
    parser.add_argument('--ft_lambda_reward', type=float, default=None)
    parser.add_argument('--ft_lr', type=float, default=None)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)

    cfg.example = args.example
    cfg.example_hint = args.example_hint
    cfg.no_plot = args.no_plot
    cfg.replication = args.replication
    cfg.vis = args.vis
    cfg.optimize = args.optimize
    cfg.mld_path = args.mld_path
    if args.spm_path is not None:
        cfg.spm_path = args.spm_path.replace('.pth', '')
    if args.mld_path is not None:
        cfg.mld_path = args.mld_path.replace('.ckpt', '')
    cfg.ft_type = args.ft_type
    cfg.ft_m = args.ft_m

    cfg.ft_prob = args.ft_prob
    cfg.ft_t = args.ft_t
    cfg.ft_k = args.ft_k
    cfg.ft_skip = args.ft_skip
    cfg.ft_reverse = args.ft_reverse
    cfg.ft_custom = args.ft_custom
    cfg.ft_lambda_reward = args.ft_lambda_reward
    cfg.ft_lr = args.ft_lr
    cfg.ft_dy = args.ft_dy
    return cfg



def parse_args_RM() -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="The main config file")

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate, default=1e-4')
    parser.add_argument('--NoiseThr', type=float, default=5, help='')
    parser.add_argument('--CLThr', type=float, default=0.9, help='')
    parser.add_argument('--CLTemp', type=float, default=0.1, help='')
    parser.add_argument('--step_aware', type=str, default='M0T0', help='')
    parser.add_argument('--maxT', type=int, default=1000, help='')
    
    parser.add_argument('--lambda_t2m', type=float, default=0, help='')
    parser.add_argument('--lambda_m2m', type=float, default=0, help='')
    parser.add_argument('--finetune', type=str, default=None, help='')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg_root = os.path.dirname(args.cfg)
    cfg_model = get_module_config(cfg.model, cfg.model.target, cfg_root)
    cfg = OmegaConf.merge(cfg, cfg_model)
    
    cfg.lr = args.lr
    cfg.NoiseThr = args.NoiseThr
    cfg.CLThr = args.CLThr
    cfg.CLTemp = args.CLTemp
    cfg.maxT = args.maxT
    cfg.step_aware = args.step_aware
    cfg.lambda_t2m = args.lambda_t2m
    if 'pth' not in args.finetune:
        cfg.finetune = None
    else:
        cfg.finetune = args.finetune
    cfg.lambda_m2m = args.lambda_m2m
    assert cfg.step_aware in ['M0T0', 'M1T0', 'M0T1', 'M1T1'], 'Error Mode'
    assert cfg.maxT in [i for i in range(1001)], 'Error MaxT'
    return cfg

