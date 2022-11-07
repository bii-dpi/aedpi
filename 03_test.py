import os
import argparse
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


BATCH_SIZE = 128
SEED = 12345


def get_recall(precisions, recalls, prec_val):
    precisions = np.abs(np.array(precisions) - prec_val)

    return str(recalls[np.argmin(precisions)])


def get_predictions(direction):
    _, testing_dl = get_dataloaders(direction, SEED, BATCH_SIZE)

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(f"models/classifier_{direction}.pt", map_location="cpu"))

    all_predictions = []
    all_ys = []
    for proteins, ligands, y in testing_dl:
        with torch.no_grad():
            if proteins.shape[0] == 1:
                continue
            proteins, ligands, y = (proteins.to(device),
                                    ligands.to(device),
                                    y.to(device))
            predictions = classifier(proteins, ligands, decode=False)

            all_predictions.append(predictions.flatten().detach().cpu())
            all_ys.append(y.float().detach().cpu())

    return np.concatenate(all_predictions), np.concatenate(all_ys)


def get_ef(y, prec_val, num_pos):
    len_to_take = int(len(y) * prec_val)

    return str(np.sum(y[:len_to_take]) / num_pos)


def get_log_auc(predicted, y, num_pos):
    prec_vals = np.linspace(.0001, 1, 10000)
    recalls = []
    for prec_val in prec_vals:
        recalls.append(float(get_ef(y, prec_val, num_pos)))

    return str(np.trapz(y=recalls, x=np.log10(prec_vals) / 3, dx=1/30))


def evaluate(direction):
    predicted, y = get_predictions(direction)

    auc = roc_auc_score(y, predicted)
    aupr = average_precision_score(y, predicted)

    precisions, recalls, _ = \
        precision_recall_curve(y, predicted)

    sorted_indices = np.argsort(predicted)[::-1]
    y = y[sorted_indices]
    predicted = predicted[sorted_indices]
    num_pos = np.sum(y)

    results = [str(auc), str(aupr), get_log_auc(predicted, y, num_pos)]
    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        results.append(get_recall(precisions, recalls, prec_val))

    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        results.append(get_ef(y, prec_val, num_pos))

    return f"{direction},{','.join(results)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('CUDA', type=int)
    args = vars(parser.parse_args())
    CUDA = args["CUDA"]
    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')

    directions = ["dtb", "btd"]

    rows = []
    for direction in directions:
        rows.append(evaluate(direction))

    with open(f"results/{CUDA}", "w") as f:
        f.write("\n".join(rows))


