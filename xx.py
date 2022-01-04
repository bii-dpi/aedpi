import pickle
import torch
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from progressbar import progressbar

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from random import randint

from ligandvae import AE, device


torch.manual_seed(123456)


class LigandDataset(Dataset):
    def __init__(self):
        self.all_fingerprints = np.load("data/all_fingerprints.npy")
        self.all_smiles = pd.read_pickle("data/all_smiles.pkl")

    def __len__(self):
        return len(self.all_fingerprints)

    def __getitem__(self, idx):
        x = self.all_fingerprints[idx]
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()

        return self.all_smiles[idx], x


dataset = LigandDataset()


# Device configuration
bs = 10000
# Load Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

ae = AE().to(device)
ae.load_state_dict(torch.load("models/ligand_ae.pt", map_location="cpu"))

ligand_dict = {}
for smiles, images in progressbar(dataloader):
    images = images.to(device)
    encoded = ae.encode(images)
    encoded = encoded.detach().cpu()
    for i in range(len(smiles)):
        ligand_dict[smiles[i]] = encoded[i]

"""
import sys
print(sys.getsizeof(ligand_dict))

ligand_dict = [(smiles, repr_) for smiles, repr_ in ligand_dict.items()]
with open("data/ligand_dict.pkl", "wb") as f:
    pickle.dump(ligand_dict, f)
"""
"""
with open("data/ligand_dict.pkl", "wb") as f:
    pickle.dump(ligand_dict, f)
"""


