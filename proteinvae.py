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


# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=2048, z_dim=512):
        super(VAE, self).__init__()
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
        self.fc2 = nn.Linear(h_dim, z_dim)
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

