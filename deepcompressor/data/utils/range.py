import torch

def absmax(tensor: torch.Tensor, reduce_dim: int):
    # call amin()/amax() before calling abs(), in order to optimize memory usage.
    t_minabs = tensor.amin(dim=reduce_dim, keepdim=True).abs()
    t_maxabs = tensor.amax(dim=reduce_dim, keepdim=True).abs()
    return torch.maximum(t_minabs, t_maxabs)