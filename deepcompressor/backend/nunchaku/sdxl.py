import torch

from .common import convert_to_nunchaku_transformer_block_state_dict, update_state_dict


def _get_sdxl_transformer_block_names(state_dict):
    transformer_block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if ".transformer_blocks." in param_name:
            if param_name.startswith("up_blocks") or param_name.startswith("down_blocks"):
                transformer_block_names.add(".".join(param_name.split(".")[:6]))
            elif param_name.startswith("mid_block"):
                transformer_block_names.add(".".join(param_name.split(".")[:5]))
            else:
                raise ValueError(f"Unknown block name: {param_name}")
        else:
            other[param_name] = state_dict[param_name]
    # all the numbers in sdxl state dict are single-digit, so there's no need to convert to int for sorting.
    transformer_block_names = sorted(transformer_block_names, key=lambda x: tuple(x.split(".")))
    return transformer_block_names, other


def convert_to_nunchaku_sdxl_state_dicts(
    model_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block_names, other = _get_sdxl_transformer_block_names(model_dict)
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        d = convert_to_nunchaku_transformer_block_state_dict(
            state_dict=model_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            block_name=block_name,
            local_name_map={
                "norm1": "norm1",
                "attn1.to_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
                "attn1.to_out.0": "attn1.to_out.0",
                "norm2": "norm2",
                "attn2.to_q": "attn2.to_q",
                "attn2.to_k": "attn2.to_k",
                "attn2.to_v": "attn2.to_v",
                "attn2.to_out.0": "attn2.to_out.0",
                "norm3": "norm3",
                "ff.net.0.proj": "ff.net.0.proj",
                "ff.net.2": "ff.net.2"
            },
            smooth_name_map={
                "attn1.to_qkv": "attn1.to_q",
                "attn1.to_out.0": "attn1.to_out.0",
                "attn2.to_q": "attn2.to_q",
                "attn2.to_out.0": "attn2.to_out.0",
                "ff.net.0.proj": "ff.net.0.proj",
                "ff.net.2": "ff.net.2",
            },
            branch_name_map={
                "attn1.to_qkv": "attn1.to_q",
                "attn1.to_out.0": "attn1.to_out.0",
                "attn2.to_q": "attn2.to_q",
                "attn2.to_out.0": "attn2.to_out.0",
                "ff.net.0.proj": "ff.net.0.proj",
                "ff.net.2": "ff.net.2",
            },
            convert_map={ 
                "attn1.to_qkv": "linear",
                "attn1.to_out.0": "linear",
                "attn2.to_q": "linear",
                "attn2.to_out.0": "linear",
                "ff.net.0.proj": "linear",
                "ff.net.2": "linear",
            },
            float_point=float_point,
        )
        update_state_dict(converted, d, prefix=block_name,)
    return converted, other
