# import gguf
# import numpy as np
# import torch
# np.set_printoptions(precision=4)
# torch.manual_seed(0)
# torch_iuput = torch.randn(1, 1, 1, 729, 1152)
# # torch_iuput = torch.randn(1, 1, 3, 20, 20)
# torch.save(torch_iuput, 'torch_input.pt')
# # print('torch_iuput\n', torch_iuput)

# ggml_input = torch_iuput.squeeze(dim=0).numpy()
# print(f'ggml_input:{ggml_input.shape}\n', ggml_input)

# w = torch.load('w.pt').numpy()
# print(f'w:{w.shape}\n', w)

# ln_w = torch.load('ln_w.pt').detach().numpy()
# print(f'ln_w:{ln_w.shape}\n', ln_w)

# ln_b = torch.load('ln_b.pt').detach().numpy()
# print(f'ln_b:{ln_b.shape}\n', ln_b)

# gguf_writer = gguf.GGUFWriter(path='ggml_input.gguf', arch='ggml_input')
# gguf_writer.add_tensor("data", ggml_input)
# gguf_writer.add_tensor("w", w)
# gguf_writer.add_tensor("ln_w", ln_w)
# gguf_writer.add_tensor("ln_b", ln_b)
# gguf_writer.write_header_to_file()
# gguf_writer.write_kv_data_to_file()
# gguf_writer.write_tensors_to_file()
# gguf_writer.close()

import gguf
import numpy as np
import torch

vision_attn_masks = torch.load('./test_data/vision_attn_masks.pt').numpy()
print(f'vision_attn_masks:{vision_attn_masks.shape}\n', vision_attn_masks)


gguf_writer = gguf.GGUFWriter(path='vision_attn_masks.gguf', arch='vision_attn_masks')
gguf_writer.add_tensor("data", vision_attn_masks)
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()



# torch.manual_seed(0)
# torch_iuput = torch.randn(1, 1, 1, 729, 1152)
# torch.save(torch_iuput, 'torch_input.pt')
# print('torch_iuput\n', torch_iuput[0,0,0, :3, :5])

# # ggml_input = torch.permute(torch_iuput, (4, 3, 2, 1, 0)).squeeze(dim=-1).numpy()
# # print('ggml_input\n', ggml_input[:5, :3, 0, 0])
# ggml_input = torch_iuput.squeeze(dim=0).numpy()
# print(f'ggml_input:{ggml_input.shape}\n', ggml_input[0,0, :5, :5])

# gguf_writer = gguf.GGUFWriter(path='ggml_input.gguf', arch='ggml_input')
# gguf_writer.add_tensor("data", ggml_input)
# gguf_writer.write_header_to_file()
# gguf_writer.write_kv_data_to_file()
# gguf_writer.write_tensors_to_file()
# gguf_writer.close()