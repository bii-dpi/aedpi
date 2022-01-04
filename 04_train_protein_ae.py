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



torch.manual_seed(12345)

if not os.path.exists('models'):
    os.mkdir('models')


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



"""
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
"""
dataset = ProteinDataset()

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
bs = 16
bs = 154
# Load Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1, 1)


class AE(nn.Module):
    def __init__(self, image_channels=1, h_dim=2048, z_dim=512):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose3d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, image_channels, kernel_size=6, stride=2),
            nn.ReLU(),
        )

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc1(h)
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z

ae = AE().to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

def loss_fn(recon_x, x):
    BCE = F.mse_loss(recon_x, x, size_average=False)

    return BCE

epochs = 1316
global_min = 10000000
best_epoch = -1
for epoch in range(epochs):
    total_loss = 0
    for idx, images in enumerate(dataloader):
        images = images.to(device)
        recon_images = ae(images)
        loss = loss_fn(recon_images, images)
        total_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1,
                            epochs, total_loss/154)
    print(to_print)
    if total_loss/154 < global_min:
        global_min = total_loss/154
        best_epoch = epoch
        torch.save(ae.state_dict(), 'models/protein_ae.pt')
    print(global_min, best_epoch)

