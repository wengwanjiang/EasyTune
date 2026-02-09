import copy
from typing import Optional

import torch.nn as nn
import torch



ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "glu": nn.GLU(),
}


def get_clone(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_activation_fn(act_fn: Optional[str] = None) -> nn.Module:
    if act_fn is None:
        return nn.Identity()
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module



def lengths_to_mask(lengths_list: list[int],
                    device: torch.device,
                    max_len: int = None) -> torch.Tensor:
    lengths = torch.as_tensor(lengths_list, dtype=torch.long, device=device)
    # 只有当用户没传入 max_len 时，才取 lengths 的最大值
    if max_len is None:
        max_len = int(lengths.max().item())
    # arange + 比较：对短的补 False，对长的截断
    mask = (torch.arange(max_len, device=device)
            .unsqueeze(0)
            .expand(len(lengths_list), max_len)
           ) < lengths.unsqueeze(1)
    return mask

def collate_tensors(batch: list) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate_motion_only(batch: list) -> dict:
    batch = {
        "motion": collate_tensors([torch.tensor(b[0]).float() for b in batch]),
        "length": [b[1] for b in batch]
    }
    return batch
