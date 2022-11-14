import os
import torch
from torch import nn


H_DIM = 512
H_DIM = 256

if not os.path.exists('models'):
    os.mkdir('models')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=H_DIM):
        return input.view(input.size(0), size, 1, 1, 1)


class Classifier(nn.Module):
    def __init__(self, h_dim=H_DIM, z_dim=512):
        super(Classifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),

            nn.ConvTranspose3d(H_DIM, 32, kernel_size=6, stride=2),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=6, stride=2),
            nn.ReLU(),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, volume):
        flattened = self.encoder(volume)
        encoded = self.fc1(flattened)

        return encoded

    def decode(self, encoded):
        flattened = self.fc2(encoded)
        decoded = self.decoder(flattened)

        return decoded

    def forward(self, volume, decode):
        encoded = self.encode(volume)

        if not decode:
            return self.fcnn(encoded)
        else:
            return self.decode(encoded)

