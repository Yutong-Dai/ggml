import torch
from gguf import *
import re
import numpy as np
import os

def _replace_attn_layer(key, value):
    # Check for the special case first
    if re.match(r'layers\.(\d+)\.0\.to_kv\.weight', key):
        idx = re.search(r'layers\.(\d+)\.0\.to_kv\.weight', key).group(1)
        KVweight = value.chunk(2, dim=0)
        return {f'blk.{idx}.attn.to_k.weight': KVweight[0],
                f'blk.{idx}.attn.to_v.weight': KVweight[1]
                }
    
    # Apply general replacements for other patterns
    # Define the replacement patterns
    patterns = [
        (r'layers\.(\d+)\.0\.norm_media\.(weight|bias)', r'blk.\1.attn.norm_media.\2'),
        (r'layers\.(\d+)\.0\.norm_latents\.(weight|bias)', r'blk.\1.attn.norm_latents.\2'),
        (r'layers\.(\d+)\.0\.to_q\.(weight)', r'blk.\1.attn.to_q.\2'),
        (r'layers\.(\d+)\.0\.to_out\.(weight)', r'blk.\1.attn.to_out.\2'),
        (r'layers\.(\d+)\.1\.0\.(weight|bias)', r'blk.\1.ffn.ln.\2'),
        (r'layers\.(\d+)\.1\.1\.weight', r'blk.\1.ffn.linear_up.weight'),
        (r'layers\.(\d+)\.1\.3\.weight', r'blk.\1.ffn.linear_down.weight'),
    ]
    for pattern, replacement in patterns:
        key = re.sub(pattern, replacement, key)
    
    return {key: value}

def replace_tensor_name_xgenmm_projector(ckpt):
    identifier = 'perceiver_resampler.'
    new_state_dict = {}
    for k, v in ckpt.items():
        # handel the layer
        if 'layers' in k:
            new_kvs = _replace_attn_layer(k, v)
            for new_k, new_v in new_kvs.items():
                new_state_dict[identifier+new_k] = new_v
        elif k == 'norm.weight':
            new_k = 'ln.weight'
            new_state_dict[identifier+new_k] = v
        elif k == 'norm.bias':
            new_k = 'ln.bias'
            new_state_dict[identifier+new_k] = v  
        else:
            new_state_dict[identifier+k] = v
    return new_state_dict     

if __name__ == '__main__':
    use_f32 = True    
    DIR = '/export/share/yutong/xgenmm/llamacpp_wd/siglip_kosmos_phi3_4k_instruct'
    ckpt = torch.load(DIR + '/xgenmm.projector')
    projector = replace_tensor_name_xgenmm_projector(ckpt)
    
    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    #
    # map from ftype to string
    ftype_str = ["f32", "f16"]
    ftype = 1
    if use_f32:
        ftype = 0
    output_dir = './'
    fname_out = os.path.join(output_dir, f"projector-{ftype_str[ftype]}.gguf")
    fout = GGUFWriter(path=fname_out, arch="PercevierResampler")
    ftype_cur = ftype
    for name, tensor in projector.items():
        tensor = tensor.squeeze().numpy()
        if ftype_cur == 1:
            if 'ln.bias' in name or 'ln.weight' in name:
                tensor = tensor.astype(np.float32)
                ftype_cur = 0
                print(f'❗ {name} is set to np.float32')
            else:
                tensor = tensor.astype(np.float16)
                ftype_cur = 1
                print(f'❗ {name} is set to np.float16')
        else:
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
                print(f'❗ {name} is set to np.float32')
                ftype_cur = 0

        print(f"{name} - {ftype_str[ftype_cur]} - shape = {tensor.shape}")
        fout.add_tensor(name, tensor)
    print("🟢 Projector tensors added\n")
    
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()

    print("🟢 Done. Output file: " + fname_out)