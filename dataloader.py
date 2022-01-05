import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ComplexDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.set_protein_dict()
        self.set_ligand_dict()

    def set_protein_dict(self):
        def process_protein(path):
            protein = np.load(path)
            protein = np.expand_dims(protein, axis=0)
            return torch.from_numpy(protein).float()

        grid_list = [process_protein(f"data/grids/{fname}")
                     for fname in os.listdir("data/grids/")
                     if fname.endswith("_processed.npy")]

        pdb_id_list = [fname.split("_")[0]
                       for fname in os.listdir("data/grids/")
                       if fname.endswith("_processed.npy")]

        self.protein_dict = dict(zip(pdb_id_list, grid_list))

    def set_ligand_dict(self):
        def process_ligand(fingerprint):
#            fingerprint = np.expand_dims(fingerprint, axis=0)
            return torch.from_numpy(fingerprint).float()

        all_fingerprints = [process_ligand(ligand) for ligand in
                            np.load("data/all_fingerprints.npy")]
        all_smiles = pd.read_pickle("data/all_smiles.pkl")

        self.ligand_dict = dict(zip(all_smiles, all_fingerprints))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ligand = self.ligand_dict[self.examples[idx][0]]
        protein = self.protein_dict[self.examples[idx][1]]
        label = self.examples[idx][2]

        return protein, ligand, label


def read_examples(path):
    with open(path, "r") as f:
        examples = [line.split()[:2] for line in f.readlines()]
    examples = [[line[0], line[1][:4]] for line in examples]

    if path.endswith("actives"):
        return [line + [1.] for line in examples]
    else:
        return [line + [0.] for line in examples]


def get_dataloaders(direction, resample, seed, batch_size):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if direction == "dztbz":
        dataset = "DUDE"
    else:
        dataset = "BindingDB"

    actives = \
        read_examples(f"../get_data/{dataset}/{dataset.lower()}_actives")
    decoys = \
        read_examples(f"../get_data/{dataset}/{dataset.lower()}_zs_decoys")

    if resample:
        np.random.shuffle(decoys)
        decoys = decoys[:len(actives)]

    examples = actives + decoys
    np.random.shuffle(examples)

    training_examples = examples[:int(len(examples) * 0.8)]
    validation_examples = examples[int(len(examples) * 0.8):]

    training_ds = ComplexDataset(training_examples)
    validation_ds = ComplexDataset(validation_examples)

    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)

    return training_dl, validation_dl

