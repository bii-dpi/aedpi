import torch
import pickle

import numpy as np
import pandas as pd

from scipy.ndimage import zoom
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


RESOLUTION = 1
LENGTH = 16.


pdb_ids = \
    list(pd.read_pickle(f"../get_data/BindingDB/sequence_to_id_map.pkl").values())
pdb_ids += \
    list(pd.read_pickle(f"../get_data/DUDE/sequence_to_id_map.pkl").values())
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != "5YZ0_B"]


elements_list = ['C', 'N', 'O', 'P', 'S', 'F', 'CL', 'BR', 'I']
elements_dict = dict(zip(elements_list, range(1, len(elements_list) + 1)))


def encode_elements(elements):
    encoded = []
    for element in elements:
        try:
           encoded.append(elements_dict[element])
        except:
           encoded.append(10)

    return encoded


def get_grid(input_list):
    def get_range(col):
        values = coords[:, col].flatten()
        return round(np.max(values))

    input_list = np.vstack(input_list)

    coords = input_list[:, :-1].astype(float)
    coords -= np.min(coords, axis=0)

    elements = input_list[:, -1]
    elements = encode_elements(elements)

    coords = np.rint(coords).astype(np.int16)
    grid = np.zeros([get_range(col) for col in range(3)])

    for i in range(len(coords)):
        x, y, z = coords[i, :] - 1
        grid[x][y][z] = max(grid[x][y][z], elements[i])


    grid = zoom(grid, LENGTH / np.array(grid.shape))
    grid = np.expand_dims(grid, axis=0)

    return torch.from_numpy(grid).float()


def write_grid(pdb_id):
    ligand_dict = \
        pd.read_pickle(f"../graphmake/proc_ligands/{pdb_id}.pkl")

    grids_dict = dict()
    for smiles in progressbar(ligand_dict):
        grids_dict[smiles] = get_grid(ligand_dict[smiles][0])

    with open(f"data/grids/{pdb_id}_ligand_grids.pkl", "wb") as f:
        pickle.dump(grids_dict, f)


if __name__ == "__main__":
    '''
    for pdb_id in progressbar(pdb_ids):
        write_grid(pdb_id)
    '''

    with PPE() as executor:
        executor.map(write_grid, pdb_ids)

