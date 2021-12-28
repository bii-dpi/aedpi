import pickle
import torch
import torchvision
import numpy as np
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

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from random import randint

from proteinvae import VAE, device


torch.manual_seed(123456)


class ProteinDataset(Dataset):
    def __init__(self):
        self.grid_list = [f"data/grids/{fname}"
                          for fname in os.listdir("data/grids/")
                          if fname.endswith("_processed.npy")]

    def __len__(self):
        return len(self.grid_list)

    def __getitem__(self, idx):
        image = np.load(self.grid_list[idx])
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()

        return image

    def get_grid_list(self):
        return [path.split("/")[-1][:4] for path in self.grid_list]


dataset = ProteinDataset()

# Device configuration
bs = 154
# Load Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

vae = VAE().to(device)
vae.load_state_dict(torch.load('models/protein_vae.pt', map_location='cpu'))


for images in dataloader:
    images = images.to(device)
    encoded, _, _ = vae.encode(images)

protein_list = dataset.get_grid_list()
protein_dict = {}
for i in range(len(encoded)):
    protein_dict[protein_list[i]] = encoded[i]

print(encoded.shape)
print(protein_dict["3EML"].shape)

with open("data/protein_dict.pkl", "wb") as f:
    pickle.dump(protein_dict, f)


