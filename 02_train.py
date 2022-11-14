import os
import argparse
import torch
import numpy as np
from model import Classifier
from dataloader import get_dataloaders
from progressbar import progressbar
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


BATCH_SIZE = 128
SEED = 12345
GAP = 800
LR = 1e-4


def get_bce_loss(predictions, y):
    BCE = F.binary_cross_entropy(predictions.flatten(), y.float(), reduction="sum")

    return BCE


def get_mse_loss(decoded_volumes, volumes):
    MSE_volume = F.mse_loss(decoded_volumes, volumes, reduction="sum")

    return MSE_volume


def get_AUPR(all_ys, all_predictions):
    all_ys = np.concatenate(all_ys)
    all_predictions = np.concatenate(all_predictions)

    return average_precision_score(all_ys, all_predictions)


def save_trained(direction):
    '''
    if os.path.isfile(f"models/classifier_{direction}.pt"):
        return
    '''

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    training_dl, _  = get_dataloaders(direction, SEED, BATCH_SIZE)

    classifier = Classifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    epochs = 100
    #global_max, best_epoch = 0, -1
    global_max, best_epoch = 1e8, -1
    since_best = 0
    for epoch in progressbar(range(epochs)):
        for iteration, (volumes, y) in enumerate(training_dl):
            if volumes.shape[0] == 1:
                continue
            optimizer.zero_grad()
            volumes, y = (volumes.to(device),
                           y.to(device))

            if iteration % GAP == 0:
                decoded_volumes = \
                    classifier(volumes, decode=True)

                mse = get_mse_loss(decoded_volumes, volumes)
                mse.backward()
            else:
                predictions = classifier(volumes, decode=False)

                bce = get_bce_loss(predictions, y)
                bce.backward()

            optimizer.step()

    torch.save(classifier.state_dict(),
               f"models/classifier_{direction}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('CUDA', type=int)
    args = vars(parser.parse_args())
    CUDA = args["CUDA"]

    directions = ["btd", "dtb"]
    for direction in progressbar(directions):
        print(direction)
        save_trained(direction)


