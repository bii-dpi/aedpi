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


def read_examples(direction, suffix):
    with open(f"../get_data/AEDPI/data/{direction}_{suffix}", "r") as f:
        examples = [line.split() for line in f.readlines()]

    examples = [[line[0], line[1][:4], float(line[2])] for line in examples]
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

