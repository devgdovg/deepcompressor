'''printed from ZImageStruct._get_default_key_map()'''
_map = {
    'transformer_norm': {
        'lk_transformer_norm',
        'nrk_transformer_norm',
        'crk_transformer_norm'
    },
    'nrk_transformer_norm': {
        'nrk_transformer_norm'
    },
    'nrk': {
        'nrk_attn_add_qkv_proj',
        'nrk_ffn_add_up_proj',
        'nrk_ffn_up_proj',
        'nrk_ffn_add_down_proj',
        'nrk_attn_out_proj',
        'nrk_transformer_norm',
        'nrk_ffn_down_proj',
        'nrk_transformer_add_norm',
        'nrk_attn_add_out_proj',
        'nrk_attn_qkv_proj'
    },
    'transformer_add_norm': {
        'nrk_transformer_add_norm',
        'lk_transformer_add_norm',
        'crk_transformer_add_norm'
    },
    'nrk_transformer_add_norm': {
        'nrk_transformer_add_norm'
    },
    'attn': {
        'crk_attn_out_proj',
        'nrk_attn_out_proj',
        'lk_attn_out_proj',
        'lk_attn_qkv_proj',
        'crk_attn_qkv_proj',
        'nrk_attn_qkv_proj'
    },
    'nrk_attn': {
        'nrk_attn_qkv_proj',
        'nrk_attn_out_proj'
    },
    'attn_add': {
        'nrk_attn_add_qkv_proj',
        'lk_attn_add_qkv_proj',
        'crk_attn_add_out_proj',
        'crk_attn_add_qkv_proj',
        'nrk_attn_add_out_proj',
        'lk_attn_add_out_proj'
    },
    'nrk_attn_add': {
        'nrk_attn_add_qkv_proj',
        'nrk_attn_add_out_proj'
    },
    'attn_qkv_proj': {
        'lk_attn_qkv_proj',
        'crk_attn_qkv_proj',
        'nrk_attn_qkv_proj'
    },
    'nrk_attn_qkv_proj': {
        'nrk_attn_qkv_proj'
    },
    'attn_out_proj': {
        'lk_attn_out_proj',
        'crk_attn_out_proj',
        'nrk_attn_out_proj'
    },
    'nrk_attn_out_proj': {
        'nrk_attn_out_proj'
    },
    'attn_add_qkv_proj': {
        'nrk_attn_add_qkv_proj',
        'lk_attn_add_qkv_proj',
        'crk_attn_add_qkv_proj'
    },
    'nrk_attn_add_qkv_proj': {
        'nrk_attn_add_qkv_proj'
    },
    'attn_add_out_proj': {
        'lk_attn_add_out_proj',
        'crk_attn_add_out_proj',
        'nrk_attn_add_out_proj'
    },
    'nrk_attn_add_out_proj': {
        'nrk_attn_add_out_proj'
    },
    'ffn': {
        'crk_ffn_up_proj',
        'nrk_ffn_up_proj',
        'lk_ffn_up_proj',
        'nrk_ffn_down_proj',
        'crk_ffn_down_proj',
        'lk_ffn_down_proj'
    },
    'nrk_ffn': {
        'nrk_ffn_down_proj',
        'nrk_ffn_up_proj'
    },
    'ffn_add': {
        'nrk_ffn_add_up_proj',
        'crk_ffn_add_down_proj',
        'lk_ffn_add_up_proj',
        'nrk_ffn_add_down_proj',
        'crk_ffn_add_up_proj',
        'lk_ffn_add_down_proj'
    },
    'nrk_ffn_add': {
        'nrk_ffn_add_up_proj',
        'nrk_ffn_add_down_proj'
    },
    'ffn_up_proj': {
        'nrk_ffn_up_proj',
        'lk_ffn_up_proj',
        'crk_ffn_up_proj'
    },
    'nrk_ffn_up_proj': {
        'nrk_ffn_up_proj'
    },
    'ffn_down_proj': {
        'nrk_ffn_down_proj',
        'crk_ffn_down_proj',
        'lk_ffn_down_proj'
    },
    'nrk_ffn_down_proj': {
        'nrk_ffn_down_proj'
    },
    'ffn_add_up_proj': {
        'nrk_ffn_add_up_proj',
        'crk_ffn_add_up_proj',
        'lk_ffn_add_up_proj'
    },
    'nrk_ffn_add_up_proj': {
        'nrk_ffn_add_up_proj'
    },
    'ffn_add_down_proj': {
        'lk_ffn_add_down_proj',
        'nrk_ffn_add_down_proj',
        'crk_ffn_add_down_proj'
    },
    'nrk_ffn_add_down_proj': {
        'nrk_ffn_add_down_proj'
    },
    'crk_transformer_norm': {
        'crk_transformer_norm'
    },
    'crk': {
        'crk_ffn_up_proj',
        'crk_attn_out_proj',
        'crk_ffn_add_down_proj',
        'crk_attn_add_out_proj',
        'crk_transformer_add_norm',
        'crk_attn_add_qkv_proj',
        'crk_ffn_add_up_proj',
        'crk_ffn_down_proj',
        'crk_transformer_norm',
        'crk_attn_qkv_proj'
    },
    'crk_transformer_add_norm': {
        'crk_transformer_add_norm'
    },
    'crk_attn': {
        'crk_attn_qkv_proj',
        'crk_attn_out_proj'
    },
    'crk_attn_add': {
        'crk_attn_add_out_proj',
        'crk_attn_add_qkv_proj'
    },
    'crk_attn_qkv_proj': {
        'crk_attn_qkv_proj'
    },
    'crk_attn_out_proj': {
        'crk_attn_out_proj'
    },
    'crk_attn_add_qkv_proj': {
        'crk_attn_add_qkv_proj'
    },
    'crk_attn_add_out_proj': {
        'crk_attn_add_out_proj'
    },
    'crk_ffn': {
        'crk_ffn_up_proj',
        'crk_ffn_down_proj'
    },
    'crk_ffn_add': {
        'crk_ffn_add_up_proj',
        'crk_ffn_add_down_proj'
    },
    'crk_ffn_up_proj': {
        'crk_ffn_up_proj'
    },
    'crk_ffn_down_proj': {
        'crk_ffn_down_proj'
    },
    'crk_ffn_add_up_proj': {
        'crk_ffn_add_up_proj'
    },
    'crk_ffn_add_down_proj': {
        'crk_ffn_add_down_proj'
    },
    'lk_transformer_norm': {
        'lk_transformer_norm'
    },
    'lk': {
        'lk_ffn_add_up_proj',
        'lk_transformer_add_norm',
        'lk_attn_add_qkv_proj',
        'lk_transformer_norm',
        'lk_ffn_up_proj',
        'lk_attn_out_proj',
        'lk_attn_qkv_proj',
        'lk_ffn_add_down_proj',
        'lk_ffn_down_proj',
        'lk_attn_add_out_proj'
    },
    'lk_transformer_add_norm': {
        'lk_transformer_add_norm'
    },
    'lk_attn': {
        'lk_attn_out_proj',
        'lk_attn_qkv_proj'
    },
    'lk_attn_add': {
        'lk_attn_add_qkv_proj',
        'lk_attn_add_out_proj'
    },
    'lk_attn_qkv_proj': {
        'lk_attn_qkv_proj'
    },
    'lk_attn_out_proj': {
        'lk_attn_out_proj'
    },
    'lk_attn_add_qkv_proj': {
        'lk_attn_add_qkv_proj'
    },
    'lk_attn_add_out_proj': {
        'lk_attn_add_out_proj'
    },
    'lk_ffn': {
        'lk_ffn_down_proj',
        'lk_ffn_up_proj'
    },
    'lk_ffn_add': {
        'lk_ffn_add_down_proj',
        'lk_ffn_add_up_proj'
    },
    'lk_ffn_up_proj': {
        'lk_ffn_up_proj'
    },
    'lk_ffn_down_proj': {
        'lk_ffn_down_proj'
    },
    'lk_ffn_add_up_proj': {
        'lk_ffn_add_up_proj'
    },
    'lk_ffn_add_down_proj': {
        'lk_ffn_add_down_proj'
    }
}