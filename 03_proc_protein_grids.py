import os

import numpy as np

from scipy.ndimage import zoom
from progressbar import progressbar


LENGTH = 64.

raw_grid_paths = ["data/grids/" + fname for fname in os.listdir("data/grids")
                  if not fname.endswith("_processed.npy")]

"""
def get_size(path):
    return np.array(np.load(path).shape)

size = np.zeros((3,))
for path in raw_grid_paths:
    print(get_size(path))
    size += get_size(path)

size /= len(raw_grid_paths)
"""

def save_pooled_grid(path):
    arr = np.load(path)
    arr = zoom(arr, LENGTH / np.array(arr.shape))
    np.save(f"{path[:-4]}_processed.npy", arr)


for path in progressbar(raw_grid_paths):
    save_pooled_grid(path)

