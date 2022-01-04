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


torch.manual_seed(1234)

if not os.path.exists('models'):
    os.mkdir('models')


class LigandDataset(Dataset):
    def __init__(self):
        self.all_fingerprints = np.load("data/all_fingerprints.npy")

    def __len__(self):
        return len(self.all_fingerprints)

    def __getitem__(self, idx):
        x = self.all_fingerprints[idx]
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()

        return x


dataset = LigandDataset()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bs = 10000
# Load Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)


class VAE(nn.Module):
    def __init__(self, h_dim=2048, z_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

vae = VAE().to(device)
#model.load_state_dict(torch.load('models/protein_vae.pt', map_location='cpu'))
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

def loss_fn(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

epochs = 500
global_min, best_epoch = 1000000, -1
for epoch in range(epochs):
    total_loss, total_bce, total_kld = 0, 0, 0
    for idx, xs in enumerate(dataloader):
        xs = xs.to(device)
        recon_xs, mu, logvar = vae(xs)
        loss, bce, kld = loss_fn(recon_xs, xs, mu, logvar)
        total_loss += loss.data.item()
        total_bce += bce.data.item()
        total_kld += kld.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                            epochs, total_loss/len(dataset),
                            total_bce/len(dataset), total_kld/len(dataset))
    print(to_print)

    if total_loss/len(dataset) < global_min:
        global_min = total_loss/len(dataset)
        best_epoch = epoch
        torch.save(vae.state_dict(), 'models/ligand_vae.pt')

    print(global_min, best_epoch)

