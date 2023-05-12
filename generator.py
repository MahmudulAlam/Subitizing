import os
import sys
import numpy as np
from HRR.with_pytorch import normal, projection, binding

n = 6
dim = 64
bind = 0

if os.path.exists('data/key.npy'):
    print('Key already exists! terminated without generating new key-value pair.')
    sys.exit()

k = np.zeros((n, dim))  # key
v = np.zeros((n, dim))  # value
y = np.zeros((n, dim))  # key \bind value

for j, i in enumerate(range(1, 2 * n + 1, 2)):
    ki = projection(normal(shape=(1, dim), seed=1000 + i), dim=-1)
    vi = projection(normal(shape=(1, dim), seed=1000 + i + 1), dim=-1)

    yi = binding(ki, vi, dim=-1)

    k[j, :] = ki.numpy()
    v[j, :] = vi.numpy()
    y[j, :] = yi.numpy()

np.save('data/key.npy', k)
np.save('data/value.npy', v)
np.save('data/y.npy', y)
