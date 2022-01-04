import os
import torch

import numpy as np
import pandas as pd

from torch import nn
from xx import ligand_dict
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             average_precision_score,
                             precision_recall_curve)
from progressbar import progressbar

DIRECTION = "bztdz"

np.random.seed(12345)
protein_dict = pd.read_pickle("data/protein_dict.pkl")


def evaluate(predicted, y):
    auc = roc_auc_score(y, predicted)
    aupr = average_precision_score(y, predicted)

    precision = precision_score(y, np.round(predicted))
    recall = recall_score(y, np.round(predicted))

    precisions, recalls, thresholds = precision_recall_curve(y, predicted)
    thresholds = np.append(thresholds, [0.64])
    results = pd.DataFrame.from_dict({"precision": precisions, "recall": recalls,
                                      "threshold": thresholds})

    results.to_csv(f"data/{DIRECTION}.csv")

    return f"AUC: {auc}, AUPR: {aupr}, precision: {precision}, recall: {recall}"


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
        ligand = ligand_dict[self.examples[idx][0]]
        protein = protein_dict[self.examples[idx][1]]
        label = self.examples[idx][2]

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
classifier.load_state_dict(torch.load(f"models/classifier_{DIRECTION}.pt"))

dataloader = torch.utils.data.DataLoader(PairDataset(), batch_size=10000, shuffle=False)


all_ys = []
all_labels = []
for xs, labels in progressbar(dataloader):
    xs = xs.to(device)
    labels = labels.to(device)
    ys = classifier(xs)

    ys = ys.flatten().float().detach().cpu().numpy()
    all_ys.append(ys)
    labels = labels.float().detach().cpu().numpy()
    all_labels.append(labels)

print(evaluate(np.concatenate(all_ys),
               np.concatenate(all_labels)))
