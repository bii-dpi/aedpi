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

        return protein, ligand, label


def read_examples(direction, suffix):
    if suffix == "training":
        suffix += "_normal"
    with open(f"../get_data/NewData/results/text/{direction}_{suffix}", "r") as f:
        examples = [line.split() for line in f.readlines()]

    examples = [[line[0], sequence_to_id[line[1]], float(line[2])]
                for line in examples]
    old_len = len(examples)
    examples = [line for line in examples
                if line[1] in protein_grid_dict
                and line[0] in ligand_grid_dict[line[1]]]
    print(old_len - len(examples))

    actives = [line for line in examples if line[-1] == 1.0]
    decoys = [line for line in examples if line[-1] == 0.0]

    return actives, decoys


def get_dataloaders(direction, seed, batch_size):
    np.random.seed(seed)
    torch.manual_seed(seed)

    actives, decoys = read_examples(direction, "training")
    np.random.shuffle(actives)
    np.random.shuffle(decoys)

    training_actives = actives[:int(len(actives) * 0.8)]
    validation_actives = actives[int(len(actives) * 0.8):]

    training_decoys = decoys[:int(len(decoys) * 0.8)]
    validation_decoys = decoys[int(len(decoys) * 0.8):]
    training_decoys = training_decoys[:len(training_actives)]

    validation_examples = validation_actives + validation_decoys
    training_examples = training_actives + training_decoys

    np.random.shuffle(training_examples)

    actives, decoys = read_examples(direction, "testing")
    testing_examples = actives + decoys

    training_ds = ComplexDataset(training_examples)
    validation_ds = ComplexDataset(validation_examples)
    testing_ds = ComplexDataset(testing_examples)

    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)
    testing_dl = DataLoader(testing_ds, batch_size=batch_size, shuffle=True)

    return training_dl, validation_dl, testing_dl

