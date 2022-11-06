import torch
import pickle

import numpy as np
import pandas as pd

from scipy.ndimage import zoom
from progressbar import progressbar


POCKET_CUTOFF = 10.
RESOLUTION = 1
LENGTH = 16.


pdb_ids = \
    list(pd.read_pickle(f"../get_data/BindingDB/sequence_to_id_map.pkl").values())
pdb_ids += \
    list(pd.read_pickle(f"../get_data/DUDE/sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]


elements_list = ['C', 'N', 'O', 'P', 'S', 'F', 'CL', 'BR', 'I']
elements_dict = dict(zip(elements_list, range(1, len(elements_list) + 1)))


def center_coords(pocket_coords):
    pocket_coords -= np.min(pocket_coords, axis=0)

    return pocket_coords


def encode_pocket_elements(pocket_elements):
    encoded = []
    for pocket_element in pocket_elements:
        try:
           encoded.append(elements_dict[pocket_element])
        except:
           encoded.append(10)

    return encoded


def get_pocket_grid(pocket_coords, pocket_elements):
    def get_range(col):
        values = pocket_coords[:, col].flatten()
        return round(np.max(values))

    pocket_coords = np.rint(pocket_coords).astype(np.int16)
    grid = np.zeros([get_range(col) for col in range(3)])

    for i in range(len(pocket_coords)):
        x, y, z = pocket_coords[i, :] - 1
        grid[x][y][z] = max(grid[x][y][z], pocket_elements[i])

    return grid


def get_pocket_atoms(pdb_id):
    protein_pocket = \
        pd.read_pickle(f"../graphmake/proc_proteins/{pdb_id}_pocket.pkl")

    pocket_coords = [row[:-1] for row in protein_pocket]
    pocket_elements = [row[-1] for row in protein_pocket]

    return pocket_coords, pocket_elements


def write_protein_pocket_grid(pdb_id):
    pocket_coords, pocket_elements = get_pocket_atoms(pdb_id)

    pocket_coords = center_coords(pocket_coords)
    pocket_elements = encode_pocket_elements(pocket_elements)

    arr = get_pocket_grid(pocket_coords, pocket_elements)
    arr = zoom(arr, LENGTH / np.array(arr.shape))
    arr = np.expand_dims(arr, axis=0)

    with open(f"data/grids/{pdb_id}_protein_grid.pkl", "wb") as f:
        pickle.dump(torch.from_numpy(arr).float(), f)


if __name__ == "__main__":
    for pdb_id in progressbar(pdb_ids):
        write_protein_pocket_grid(pdb_id)

