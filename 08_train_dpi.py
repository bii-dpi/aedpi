import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

from xx import ligand_dict, device


torch.manual_seed(1234)
DIRECTION = "bztdz"
CUDA = 2 if DIRECTION == "bztdz" else 3

np.random.seed(12345)
protein_dict = pd.read_pickle("data/protein_dict.pkl")

device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')

class PairDataset(Dataset):
    def __init__(self):
        if DIRECTION == "dztbz":
            dataset = "DUDE"
        else:
            dataset = "BindingDB"

        actives = \
            self.read_examples(f"../get_data/{dataset}/{dataset.lower()}_actives")
        decoys = \
            self.read_examples(f"../get_data/{dataset}/{dataset.lower()}_zs_decoys")
        np.random.shuffle(decoys)
        decoys = decoys[:len(actives)]
        examples = actives + decoys
        np.random.shuffle(examples)

        self.training_examples = examples[:int(len(examples) * 0.8)]
        self.set_validation_examples(examples[int(len(examples) * 0.8):])


    def read_examples(self, path):
        with open(path, "r") as f:
            examples = [line.split()[:2] for line in f.readlines()]
        examples = [[line[0], line[1][:4]] for line in examples]
        if path.endswith("actives"):
            return [line + [1.] for line in examples]
        else:
            return [line + [0.] for line in examples]


    def __len__(self):
        return len(self.training_examples)


    def set_validation_examples(self, examples):
        # XXX: Could replace this with a dataloader, but you would need a
        # seprate dataset.
        all_x = []
        all_labels = []
        for example in examples:
            ligand = ligand_dict[example[0]]
            protein = protein_dict[example[1]]

            all_x.append(ligand * protein)
            all_labels.append(example[2])

        self.validation_examples = torch.cat(all_x).to(device)
        self.validation_labels = torch.Tensor(all_labels).to(device)


    def get_validation_examples(self):
        return self.validation_examples, self.validation_labels


    def __getitem__(self, idx):
        ligand = ligand_dict[self.training_examples[idx][0]]
        protein = protein_dict[self.training_examples[idx][1]]
        label = self.training_examples[idx][2]

        x = ligand * protein

        return x, label


dataset = PairDataset()

# Device configuration
bs = 10000
# Load Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)


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


classifier = Classifier().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()
validation_x, validation_labels = dataset.get_validation_examples()

epochs = 10000
global_min, best_epoch = 1000000, -1
for epoch in range(epochs):
    total_loss = 0
    for idx, (xs, labels) in enumerate(dataloader):
        xs = xs.to(device)
        labels = labels.to(device)
        ys = classifier(xs)
        ys = ys.flatten().float()
        loss = loss_fn(ys, labels.float())
        total_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs,
                                             total_loss/len(dataset)))

    if epoch % 5 == 0:
        with torch.no_grad():
            pred_labels = classifier(validation_x)
            curr_loss = loss_fn(pred_labels.flatten(), validation_labels)

        if  curr_loss < global_min:
            global_min = curr_loss
            best_epoch = epoch
            torch.save(classifier.state_dict(),
                       f"models/classifier_{DIRECTION}.pt")

        print(global_min, best_epoch)

