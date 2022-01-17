import pandas as pd
import numpy as np
import torch
from model import Classifier
from dataloader import get_dataloaders
from progressbar import progressbar
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             average_precision_score,
                             precision_recall_curve)
from progressbar import progressbar


DIRECTION = "bztdz"
CUDA = 0
BATCH_SIZE = 128
SEED = 12345

device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
training_dl, _ = get_dataloaders("bztdz" if DIRECTION == "dztbz" else "dztbz",
                                 False, SEED, BATCH_SIZE)

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load(f"models/classifier_{DIRECTION}.pt", map_location="cpu"))


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


all_predictions = []
all_ys = []
for proteins, ligands, y in progressbar(training_dl):
    with torch.no_grad():
        proteins, ligands, y = (proteins.to(device),
                                ligands.to(device),
                                y.to(device))
        _, _, predictions = \
            classifier(proteins, ligands)

        all_predictions.append(predictions.flatten().detach().cpu())
        all_ys.append(y.float().detach().cpu())


print(evaluate(np.concatenate(all_predictions),
               np.concatenate(all_ys)))

