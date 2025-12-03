
import torch

from .common import convert_to_nunchaku_transformer_block_state_dict, update_state_dict


def convert_to_nunchaku_flux_single_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "proj_out.linears.1.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "proj_out.linears.1"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "attn.to_q",
            "mlp_fc2": down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "ff.net.2.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "ff.net.2"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict
    context_down_proj_local_name = "ff_context.net.2.linear"
    if f"{block_name}.{context_down_proj_local_name}.weight" not in state_dict:
        context_down_proj_local_name = "ff_context.net.2"
        assert f"{block_name}.{context_down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm1.linear": "norm1.linear",
            "norm1_context.linear": "norm1_context.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "qkv_proj_context": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        convert_map={
            "norm1.linear": "adanorm_zero",
            "norm1_context.linear": "adanorm_zero",
            "qkv_proj": "linear",
            "qkv_proj_context": "linear",
            "out_proj": "linear",
            "out_proj_context": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
            "mlp_context_fc1": "linear",
            "mlp_context_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        convert_fn = convert_to_nunchaku_flux_single_transformer_block_state_dict
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_state_dict
        update_state_dict(
            converted,
            convert_fn(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other

