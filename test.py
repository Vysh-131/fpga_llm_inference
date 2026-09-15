import numpy as np
x = np.fromfile("x_blk15_ffn_down_input.bin", dtype=np.float32)
y = np.fromfile("y_blk15_ffn_down_output.bin", dtype=np.float32)
print("x:", x.shape, x.min(), x.max(), np.isnan(x).any())
print("y:", y.shape, y.min(), y.max(), np.isnan(y).any())
