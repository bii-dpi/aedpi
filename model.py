import os
import torch
from torch import nn


H_DIM = 512

if not os.path.exists('models'):
    os.mkdir('models')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=H_DIM):
        return input.view(input.size(0), size, 1, 1, 1)


class Conv3DSA(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_ch, out_ch, N):
        super().__init__()
        self.N = N
        self.C = in_ch
        self.D = 3
        self.H = 64
        self.W = 64
        self.gama = nn.Parameter(torch.tensor([0.0]))

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv3d_3 = nn.Sequential(
            # Conv3d input:N*C*D*H*W
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3d_1 = nn.Sequential(
            # Conv3d input:N*C*D*H*W
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )


    @classmethod
    def Cal_Patt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        k_x_flatten = k_x.reshape((N, C, D, 1, H * W))
        q_x_flatten = q_x.reshape((N, C, D, 1, H * W))
        v_x_flatten = v_x.reshape((N, C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 4, 3), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=4)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(N, C, D, H, W)
        return Patt


    @classmethod
    def Cal_Datt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        # k_x_transpose = k_x.permute(0, 1, 3, 4, 2)
        # q_x_transpose = q_x.permute(0, 1, 3, 4, 2)
        # v_x_transpose = v_x.permute(0, 1, 3, 4, 2)
        k_x_flatten = k_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 3, 5, 4), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=5)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(N, C, H, W, D)
        return Datt.permute(0, 1, 4, 2, 3)


    def forward(self, x):
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)

        Patt = self.Cal_Patt(k_x, q_x, v_x, self.N, self.C, self.D, self.H, self.W)
        Datt = self.Cal_Datt(k_x, q_x, v_x, self.N, self.C, self.D, self.H, self.W)

        Y = self.gama*(Patt + Datt) + x
        return Y


class Classifier(nn.Module):
    def __init__(self, h_dim=H_DIM, z_dim=512):
        super(Classifier, self).__init__()

        self.protein_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2),
            nn.ReLU(),
#            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
#            nn.BatchNorm3d(64),

            Flatten()
        )

        self.protein_fc1 = nn.Linear(h_dim, z_dim)
        self.protein_fc2 = nn.Linear(z_dim, h_dim)

        self.protein_decoder = nn.Sequential(
            UnFlatten(),

            nn.ConvTranspose3d(H_DIM, 32, kernel_size=6, stride=2),
            nn.ReLU(),
 #           nn.BatchNorm3d(32),

            nn.ConvTranspose3d(32, 1, kernel_size=6, stride=2),
            nn.ReLU(),
        )

        self.ligand_encoder = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),

            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )

        self.ligand_fc1 = nn.Linear(h_dim, z_dim)
        self.ligand_fc2 = nn.Linear(z_dim, h_dim)

        self.ligand_decoder = nn.Sequential(
            nn.Sigmoid(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 768),
            nn.Sigmoid(),
            nn.BatchNorm1d(768),

            nn.Linear(768, 1024),
            nn.Sigmoid(),
        )

        self.fcnn = nn.Sequential(
            nn.Linear(1024, 256),
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

