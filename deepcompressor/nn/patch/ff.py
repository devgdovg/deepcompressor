import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_z_image import FeedForward as ZImageFeedForward


def convert_z_image_ff(zff: ZImageFeedForward) -> FeedForward:
    assert isinstance(zff, ZImageFeedForward)
    assert zff.w1.in_features == zff.w3.in_features
    assert zff.w1.out_features == zff.w3.out_features
    assert zff.w1.out_features == zff.w2.in_features
    converted_ff = FeedForward(
        dim=zff.w1.in_features,
        dim_out=zff.w2.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=zff.w2.in_features,
        bias=False,
    ).to(dtype=zff.w1.weight.dtype, device=zff.w1.weight.device)
    
    up_proj: nn.Linear = converted_ff.net[0].proj
    down_proj: nn.Linear = converted_ff.net[2]
    with torch.no_grad():
        up_proj.weight.copy_(torch.cat([zff.w3.weight, zff.w1.weight], dim=0))
        down_proj.weight.copy_(zff.w2.weight)
    
    return converted_ff


if __name__ == "__main__":
    _dim = 50
    _hidden_dim = 100
    z_image_ff = ZImageFeedForward(dim=_dim, hidden_dim=_hidden_dim)
    with torch.no_grad():
        z_image_ff.w1.weight = Parameter(torch.randn(_hidden_dim, _dim, dtype=torch.float32))
        z_image_ff.w2.weight = Parameter(torch.randn(_dim, _hidden_dim, dtype=torch.float32))
        z_image_ff.w3.weight = Parameter(torch.randn(_hidden_dim, _dim, dtype=torch.float32))
    
    converted_ff = convert_z_image_ff(z_image_ff)
    
    x = torch.randn(_dim, dtype=torch.float32)
    
    y1 = z_image_ff(x)
    
    y2 = converted_ff(x)
    
    print(f"y1: {y1}")
    print(f"y2: {y2}")
    
    print(f"allclose: {torch.allclose(y1, y2)}")

