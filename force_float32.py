import torch

# Force PyTorch to use float32 by default
torch.set_default_dtype(torch.float32)

# Allow TensorFloat32 (TF32) operations
torch.set_float32_matmul_precision('high')

print("[INFO] PyTorch configured to use float32 precision") 