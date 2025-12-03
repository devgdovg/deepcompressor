"""Converts a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch

from .flux import convert_to_nunchaku_flux_state_dicts
from .sdxl import convert_to_nunchaku_sdxl_state_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=None, help="name of the model.")
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided, using {model_name} as the model name.")
    else:
        model_name = args.model_name
    assert model_name, "Model name must be provided."
    assert "flux" in model_name.lower() or "sdxl" in model_name.lower(), "Only Flux or SDXL models are supported."
    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")
    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}

    if "flux" in model_name.lower():
        converted_state_dict, other_state_dict = convert_to_nunchaku_flux_state_dicts(
            state_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=args.float_point,
        )
    elif "sdxl" in model_name.lower():
        converted_state_dict, other_state_dict = convert_to_nunchaku_sdxl_state_dicts(
            model_dict=state_dict,
            scale_dict=scale_dict,
            smooth_dict=smooth_dict,
            branch_dict=branch_dict,
            float_point=args.float_point,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    output_dirpath = os.path.join(args.output_root, model_name)
    os.makedirs(output_dirpath, exist_ok=True)
    safetensors.torch.save_file(converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors"))
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
    print(f"Quantized model saved to {output_dirpath}.")
