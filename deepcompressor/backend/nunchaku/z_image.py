from collections import OrderedDict
import torch

from .common import convert_to_nunchaku_transformer_block_state_dict, update_state_dict


def _print_dict(d: dict, dict_name: str):
    print(f"################# print {dict_name} ################")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(
                f"{dict_name}_key: {k} -> value tensor shape: {v.shape}, dtype: {v.dtype}")
        elif isinstance(v, OrderedDict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    print(
                        f"{dict_name}_key: {k}/{sub_k} -> tensor shape: {sub_v.shape}, dtype: {sub_v.dtype}")
                else:
                    print(
                        f"{dict_name}_key: {k}/{sub_k} -> value type: {type(sub_v)}, {sub_v}")
        else:
            print(f"{dict_name}_key: {k} -> value type: {type(v)}, {v}")
    print("\n")


def _replace_lora_and_smooth_key(transformer_block_state_dict: dict):
    replaced = {}
    for k, v in transformer_block_state_dict.items():
        if ".lora_down" in k:
            new_k = k.replace(".lora_down", ".proj_down")
        elif ".lora_up" in k:
            new_k = k.replace(".lora_up", ".proj_up")
        elif ".smooth_orig" in k:
            new_k = k.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in k:
            new_k = k.replace(".smooth", ".smooth_factor")
        else:
            new_k = k
        replaced[new_k] = v
    return replaced
        


def z_image_transformer_block_convert(
    model_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    converted_quantized_part_names = [
        "attention.to_qkv",  # attention q,k,v
        "attention.to_out.0",  # attention to_out
        "feed_forward.net.0.proj",  # feed forward up proj
        "feed_forward.net.2",  # feed forward down proj
    ]

    def _original_name(converted_name: str):
        if "to_qkv" in converted_name:
            return [
                converted_name.replace("to_qkv", "to_q"),
                converted_name.replace("to_qkv", "to_k"),
                converted_name.replace("to_qkv", "to_v")
            ]
        else:
            return converted_name

    def _smooth_name(converted_name: str):
        if "to_qkv" in converted_name:
            return converted_name.replace("to_qkv", "to_q")
        else:
            return converted_name

    _branch_name = _smooth_name

    converted_transformer_block_state_dict = convert_to_nunchaku_transformer_block_state_dict(
        state_dict=model_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            name: _original_name(name) for name in converted_quantized_part_names
        },
        smooth_name_map={
            name: _smooth_name(name) for name in converted_quantized_part_names
        },
        branch_name_map={
            name: _branch_name(name) for name in converted_quantized_part_names
        },
        convert_map={
            name: "linear" for name in converted_quantized_part_names
        },
        float_point=float_point,
    )

    not_quantized_parts = [
        # all norm layers are not quantized.
        "attention.norm_q.weight",
        "attention.norm_k.weight",
        "attention_norm1.weight",
        "attention_norm2.weight",
        "ffn_norm1.weight",
        "ffn_norm2.weight",
        "adaLN_modulation.0.weight",
        "adaLN_modulation.0.bias",
    ]

    for part_name in not_quantized_parts:
        absolute_name = f"{block_name}.{part_name}"
        if absolute_name in model_dict:
            print(f"  - Copying {block_name} weights: {part_name}")
            converted_transformer_block_state_dict[part_name] = model_dict[absolute_name].clone().cpu()

    converted_transformer_block_state_dict = _replace_lora_and_smooth_key(converted_transformer_block_state_dict)
    return converted_transformer_block_state_dict


def convert_to_nunchaku_z_image_state_dicts(
    model_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
    skip_refiners: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    _print_dict(model_dict, "model_dict")
    _print_dict(scale_dict, "scale_dict")
    _print_dict(smooth_dict, "smooth_dict")
    _print_dict(branch_dict, "branch_dict")

    transformer_block_names: set[str] = set()
    others: dict[str, torch.Tensor] = {}

    if skip_refiners:
        transfomer_block_name_prefix = ("layers.",)
    else:
        transfomer_block_name_prefix = (
            "noise_refiner.", "context_refiner.", "layers.")
    for param_name in model_dict.keys():
        if param_name.startswith(transfomer_block_name_prefix):
            block_name = ".".join(param_name.split(".")[:2])
            transformer_block_names.add(block_name)
        else:
            others[param_name] = model_dict[param_name]

    transformer_block_names = sorted(transformer_block_names, key=lambda x: (
        x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(transformer_block_names)} transformer blocks...")
    converted_state_dict: dict[str, torch.Tensor] = {}
    for b_name in transformer_block_names:
        converted_tranzformer_block = z_image_transformer_block_convert(
            model_dict=model_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            block_name=b_name,
            float_point=float_point,
        )
        update_state_dict(
            converted_state_dict,
            converted_tranzformer_block,
            prefix=b_name,
        )
    
    _print_dict(converted_state_dict, "converted_state_dict")
    _print_dict(others, "others")
    
    return converted_state_dict, others
