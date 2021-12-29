import os
import torch

import numpy as np
import pandas as pd

from torch import nn
from xx import ligand_dict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DIRECTION = "bztdz"

np.random.seed(12345)
protein_dict = pd.read_pickle("data/protein_dict.pkl")


class PairDataset(Dataset):
    def __init__(self):
        if DIRECTION == "dztbz":
            dataset = "BindingDB"
        else:
            dataset = "DUDE"

        actives = \
            self.read_examples(f"../get_data/{dataset}/{dataset.lower()}_actives")
        decoys = \
            self.read_examples(f"../get_data/{dataset}/{dataset.lower()}_zs_decoys")
        np.random.shuffle(decoys)
        decoys = decoys[:len(actives)]
        self.examples = actives + decoys


    def read_examples(self, path):
        with open(path, "r") as f:
            examples = [line.split()[:2] for line in f.readlines()]
        examples = [[line[0], line[1][:4]] for line in examples]
        if path.endswith("actives"):
            return [line + [1.] for line in examples]
        else:
            return [line + [0.] for line in examples]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        ligand = ligand_dict[self.training_examples[idx][0]]
        protein = protein_dict[self.training_examples[idx][1]]
        label = self.training_examples[idx][2]

        x = ligand * protein

        return x, label


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fwd = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                   )


    def forward(self, x):
        return self.fwd(x)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classifier = Classifier().to(device)

dataloader = torch.utils.data.DataLoader(PairDataset(), batch_size=10000, shuffle=False)


for xs, labels in progressbar(dataloader):
    xs = xs.to(device)
    labels = labels.to(device)
    ys = classifier(xs)
    ys = ys.flatten().float()
    labels = labels.float()
    loss = loss_fn(ys, labels)
    total_loss += loss.data.item()


