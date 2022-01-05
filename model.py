import os
import torch
from torch import nn

if not os.path.exists('models'):
    os.mkdir('models')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1, 1)


class Classifier(nn.Module):
    def __init__(self, h_dim=2048, z_dim=512):
        super(Classifier, self).__init__()

        self.protein_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.protein_fc1 = nn.Linear(h_dim, z_dim)
        self.protein_fc2 = nn.Linear(z_dim, h_dim)

        self.protein_decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose3d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=6, stride=2),
            nn.ReLU(),
        )

        self.ligand_encoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )

        self.ligand_fc1 = nn.Linear(h_dim, z_dim)
        self.ligand_fc2 = nn.Linear(z_dim, h_dim)

        self.ligand_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, proteins, ligands):
        flattened_proteins = self.protein_encoder(proteins)
        encoded_proteins = self.protein_fc1(flattened_proteins)

        flattened_ligands = self.ligand_encoder(ligands)
        encoded_ligands = self.ligand_fc1(flattened_ligands)

        return encoded_proteins, encoded_ligands

    def decode(self, encoded_proteins, encoded_ligands):
        flattened_proteins = self.protein_fc2(encoded_proteins)
        decoded_proteins = self.protein_decoder(flattened_proteins)

        flattened_ligands = self.ligand_fc2(encoded_ligands)
        decoded_ligands = self.ligand_decoder(flattened_ligands)

        return decoded_proteins, decoded_ligands

    def forward(self, proteins, ligands):
        encoded_proteins, encoded_ligands = self.encode(proteins, ligands)
        predictions = self.fcnn(torch.cat((encoded_proteins, encoded_ligands),
                                          dim=1))
        decoded_proteins, decoded_ligands = \
            self.decode(encoded_proteins, encoded_ligands)

        return decoded_proteins, decoded_ligands, predictions

