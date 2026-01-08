import json
import torch
from pathlib import Path

from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from safetensors.torch import load_file


def patch_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    replaced = {}
    for key, value in state_dict.items():
        patched_key = key.replace("model.diffusion_model.", "")
        if "attention.qkv" in patched_key:
            to_q, to_k, to_v = torch.chunk(value, 3, dim=0)
            replaced[patched_key.replace('qkv', 'to_q')] = to_q
            replaced[patched_key.replace('qkv', 'to_k')] = to_k
            replaced[patched_key.replace('qkv', 'to_v')] = to_v
        elif "attention.out" in patched_key:
            replaced[patched_key.replace("out", "to_out.0")] = value
        elif "attention.q_norm" in patched_key:
            replaced[patched_key.replace("q_norm", "norm_q")] = value
        elif "attention.k_norm" in patched_key:
            replaced[patched_key.replace("k_norm", "norm_k")] = value
        elif "final_layer" in patched_key:
            replaced[patched_key.replace("final_layer", "all_final_layer.2-1")] = value
        elif "x_embedder" in patched_key:
            replaced[patched_key.replace("x_embedder", "all_x_embedder.2-1")] = value
        elif "norm_final" in patched_key:
            # `norm_final` is not used in Z-Image Turbo
            continue
        else:
            replaced[patched_key] = value
    return replaced


def load_customized_z_image_transformer(path: str, dtype: str | torch.dtype):
    transformer_config = json.load(open(f"{path}/config.json", "r"))["transformer_config"]
    with torch.device("meta"):
            transformer = ZImageTransformer2DModel.from_config(transformer_config).to(dtype)
    state_dict = load_file(f"{path}/transformer.safetensors", device="cpu")
    state_dict = patch_transformer_state_dict(state_dict)
    transformer.load_state_dict(state_dict, assign=True)
    return transformer


def patch_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    replaced = {}
    for key, value in state_dict.items():
        if "decoder.norm_out" in key or "encoder.norm_out" in key:
            replaced[key.replace(".norm_out.", ".conv_norm_out.")] = value
        elif "decoder.mid" in key or "encoder.mid" in key:
            patched_key = key.replace(".mid.", ".mid_block.")
            if "attn_1" in patched_key:
                patched_key = patched_key.replace("attn_1", "attentions.0")
                if "attentions.0.k" in patched_key:
                    patched_key = patched_key.replace("attentions.0.k", "attentions.0.to_k")
                    if "to_k.weight" in patched_key:
                        replaced[patched_key] = value.squeeze()
                    else:
                        replaced[patched_key] = value
                elif "attentions.0.norm" in patched_key:
                    replaced[patched_key.replace("attentions.0.norm", "attentions.0.group_norm")] = value
                elif "attentions.0.proj_out" in patched_key:
                    replaced[patched_key.replace("attentions.0.proj_out", "attentions.0.to_out.0")] = value.squeeze()
                elif "attentions.0.q" in patched_key:
                    patched_key = patched_key.replace("attentions.0.q", "attentions.0.to_q")
                    if "to_q.weight" in patched_key:
                        replaced[patched_key] = value.squeeze()
                    else:
                        replaced[patched_key] = value
                elif "attentions.0.v" in patched_key:
                    patched_key = patched_key.replace("attentions.0.v", "attentions.0.to_v")
                    if "to_v.weight" in patched_key:
                        replaced[patched_key] = value.squeeze()
                    else:
                        replaced[patched_key] = value
                else:
                    raise ValueError(f"Unexpected key in VAE state dict: {key}")
            elif "block_1" in patched_key:
                replaced[patched_key.replace("block_1", "resnets.0")] = value
            elif "block_2" in patched_key:
                replaced[patched_key.replace("block_2", "resnets.1")] = value
            else:
                raise ValueError(f"Unexpected key in VAE state dict: {key}")
        elif "decoder.up" in key:
            if "decoder.up.0" in key:
                patched_key = key.replace("decoder.up.0", "decoder.up_blocks.3")
            elif "decoder.up.1" in key:
                patched_key = key.replace("decoder.up.1", "decoder.up_blocks.2")
            elif "decoder.up.2" in key:
                patched_key = key.replace("decoder.up.2", "decoder.up_blocks.1")
            elif "decoder.up.3" in key:
                patched_key = key.replace("decoder.up.3", "decoder.up_blocks.0")
            else:
                raise ValueError(f"Unexpected key in VAE state dict: {key}")
            if ".block." in patched_key:
                patched_key = patched_key.replace(".block.", ".resnets.")
                if "nin_shortcut" in patched_key:
                    patched_key = patched_key.replace("nin_shortcut", "conv_shortcut")
            elif ".upsample." in patched_key:
                patched_key = patched_key.replace(".upsample.", ".upsamplers.0.")
            else:
                raise ValueError(f"Unexpected key in VAE state dict: {key}")
            replaced[patched_key] = value
        elif "encoder.down" in key:
            patched_key = key.replace("encoder.down.", "encoder.down_blocks.")
            if ".block." in patched_key:
                patched_key = patched_key.replace(".block.", ".resnets.")
                if "nin_shortcut" in patched_key:
                    patched_key = patched_key.replace("nin_shortcut", "conv_shortcut")
            elif ".downsample." in patched_key:
                patched_key = patched_key.replace(".downsample.", ".downsamplers.0.")
            else:
                raise ValueError(f"Unexpected key in VAE state dict: {key}")
            replaced[patched_key] = value
        else:
            replaced[key] = value
    return replaced
    
def load_customized_z_image_vae(path: str, dtype: str | torch.dtype):
    vae_config = json.load(open(f"{path}/config.json", "r"))["vae_config"]
    with torch.device("meta"):
            vae = AutoencoderKL.from_config(vae_config).to(dtype)
    state_dict = load_file(f"{path}/ae.safetensors", device="cpu")
    state_dict = patch_vae_state_dict(state_dict)
    vae.load_state_dict(state_dict, assign=True)
    return vae

    
def build_customized_z_image_pipeline(name: str, path: str, dtype: str | torch.dtype, device: str | torch.device) -> DiffusionPipeline:
    assert name == "z-image-customized", f"Unsupported pipeline name: {name}"
    DIFFUSERS_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
    transformer = load_customized_z_image_transformer(path, dtype)
    vae = load_customized_z_image_vae(path, dtype)
    pipe = ZImagePipeline.from_pretrained(
        DIFFUSERS_REPO_ID,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    ).to(device)
    return pipe


def check_z_image_customized_path(path: str) -> bool:
    dir_path = Path(path)
    if not dir_path.is_dir():
        return False
    transformer_file = dir_path / "transformer.safetensors"
    ae_file = dir_path / "ae.safetensors"
    config_file = dir_path / "config.json"
    return transformer_file.is_file() and ae_file.is_file() and config_file.is_file()
