import os
import torch
import numpy as np
import pandas as pd
from progressbar import progressbar
from torch.utils.data import Dataset, DataLoader


pdb_ids = \
    list(pd.read_pickle("../get_data/BindingDB/sequence_to_id_map.pkl").values())#[:2]
pdb_ids += \
    list(pd.read_pickle("../get_data/DUDE/sequence_to_id_map.pkl").values())#[:2]
pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in ["5YZ0_B", "6WHC_R"]]
print(len(pdb_ids))


protein_grid_dict = dict()
ligand_grid_dict = dict()
for pdb_id in progressbar(pdb_ids):
    protein_grid_dict[pdb_id] = \
        pd.read_pickle(f"data/grids/{pdb_id}_protein_grid.pkl")
    ligand_grid_dict[pdb_id] = \
        pd.read_pickle(f"data/grids/{pdb_id}_ligand_grids.pkl")


sequence_to_id = pd.read_pickle("../get_data/BindingDB/sequence_to_id_map.pkl")
sequence_to_id.update(pd.read_pickle("../get_data/DUDE/sequence_to_id_map.pkl"))


class ComplexDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ligand = ligand_grid_dict[self.examples[idx][1]][self.examples[idx][0]]
        protein = protein_grid_dict[self.examples[idx][1]]
        label = self.examples[idx][2]

        return torch.cat((protein, ligand)), label


def read_examples(direction, suffix):
    with open(f"../get_data/NewData/results/text/{direction}_{suffix}", "r") as f:
        examples = [line.split() for line in f.readlines()]

    examples = [[line[0], sequence_to_id[line[1]], float(line[2])]
                for line in examples]
    old_len = len(examples)
    examples = [line for line in examples
                if line[1] in protein_grid_dict
                and line[0] in ligand_grid_dict[line[1]]]
    print(old_len - len(examples))

    return examples


def get_dataloaders(direction, seed, batch_size):
    np.random.seed(seed)
    torch.manual_seed(seed)

    training_examples = read_examples(direction, "training_normal")

    testing_examples = read_examples(direction, "testing")

    training_ds = ComplexDataset(training_examples)
    testing_ds = ComplexDataset(testing_examples)

    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    testing_dl = DataLoader(testing_ds, batch_size=batch_size, shuffle=True)

    return training_dl, testing_dl

