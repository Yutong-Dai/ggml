import gguf
import numpy as np
import os
import torch
save_dir = './data_gguf'
os.makedirs(save_dir, exist_ok=True)
A = np.array(
    [2, 8,
    5, 1,
    4, 2,
    8, 6], dtype=np.float32).reshape(4, 2)

B = np.array(
    [ 
     10, 9, 5, 
      5, 9, 4],
    dtype=np.float32).reshape(2, 3)
print(f"np.matmul(A, B) = \n{np.matmul(A, B)}")

fname_out = f"{save_dir}/mul_mat_data.gguf"
gguf_writer = gguf.GGUFWriter(fname_out, "mul_mat_data")
gguf_writer.add_tensor("A", A, raw_shape=(4, 2))
gguf_writer.add_tensor("B", B, raw_shape=(2, 3))
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()


dim0 = 2
dim1 = 3
dim2 = 4
tensor_3d = np.arange(dim0*dim1*dim2).reshape(dim0, dim1, dim2).astype(np.float32)
print(f"tensor = \n{tensor_3d}")
fname_out = f"{save_dir}/3d_tensor.gguf"
gguf_writer = gguf.GGUFWriter(fname_out, "3d_tensor")
gguf_writer.add_tensor("data", tensor_3d, raw_shape=(dim0, dim1, dim2))
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()



fname_out = f"{save_dir}/batched_mat_mul.gguf"
gguf_writer = gguf.GGUFWriter(fname_out, "batched_mat_mul")
dim0 = 2
dim1 = 3
dim2 = 4
U = np.arange(dim0*dim1*dim2).reshape(dim0, dim1, dim2).astype(np.float32)
gguf_writer.add_tensor("U", U, raw_shape=(dim0, dim1, dim2))
dim0 = 2
dim1 = 4
dim2 = 5
V = np.arange(dim0*dim1*dim2).reshape(dim0, dim1, dim2).astype(np.float32)
# permute for ggml to consume
V_tranposed = np.transpose(V, (0, 2, 1))
gguf_writer.add_tensor("V",V_tranposed, raw_shape=V_tranposed.shape)
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
print(f"U = \n{U}")
print(f"V = \n{V}")
print(f"V_tranposed = \n{V_tranposed}")
result = torch.bmm(torch.tensor(U), torch.tensor(V))
print(f"result = \n{result}")