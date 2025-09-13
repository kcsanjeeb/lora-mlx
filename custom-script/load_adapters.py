import numpy as np

# Load LoRA adapter file
adapters1 = np.load("adapters.npz")
print("Keys in adapters.npz:", adapters1.files)

# Check shapes and sizes
for key in adapters1.files:
    arr = adapters1[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024:.2f} KB")
