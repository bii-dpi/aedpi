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
GAP = 8
LR = 1e-4


def get_bce_loss(predictions, y):
    BCE = F.binary_cross_entropy(predictions.flatten(), y.float(), reduction="sum")

    return BCE


def get_mse_loss(decoded_proteins, proteins, decoded_ligands, ligands):
    MSE_protein = F.mse_loss(decoded_proteins, proteins, reduction="sum")
    MSE_ligand = F.mse_loss(decoded_ligands, ligands, reduction="sum")

    return MSE_protein + MSE_ligand


def get_AUPR(all_ys, all_predictions):
    all_ys = np.concatenate(all_ys)
    all_predictions = np.concatenate(all_predictions)

    return average_precision_score(all_ys, all_predictions)


def save_trained(direction):
    if os.path.isfile(f"models/classifier_{direction}.pt"):
        return

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    training_dl, validation_dl, _  = get_dataloaders(direction, SEED, BATCH_SIZE)

    classifier = Classifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    epochs = 100
    #global_max, best_epoch = 0, -1
    global_max, best_epoch = 1e8, -1
    since_best = 0
    for epoch in range(epochs):
        for iteration, (proteins, ligands, y) in enumerate(training_dl):
            if proteins.shape[0] == 1:
                continue
            optimizer.zero_grad()
            proteins, ligands, y = (proteins.to(device),
                                    ligands.to(device),
                                    y.to(device))

            if iteration % GAP == 0:
                decoded_proteins, decoded_ligands = \
                    classifier(proteins, ligands, decode=True)

                mse = get_mse_loss(decoded_proteins, proteins,
                                   decoded_ligands, ligands)
                mse.backward()
            else:
                predictions = classifier(proteins, ligands, decode=False)

                bce = get_bce_loss(predictions, y)
                bce.backward()

            optimizer.step()

        # Need to also do the evaluation at the very end.
        # Don't need to do patience for now.
        if epoch % 5 == 0:
            all_predictions, all_ys = [], []
            with torch.no_grad():
                total_bce = 0
                for proteins, ligands, y in validation_dl:
                    if proteins.shape[0] == 1:
                        continue
                    proteins, ligands, y = (proteins.to(device),
                                            ligands.to(device),
                                            y.to(device))
                    predictions = classifier(proteins, ligands, decode=False)
                    all_predictions.append(predictions.detach().cpu())
                    all_ys.append(y.detach().cpu())

                    total_bce += get_bce_loss(predictions, y).data.item()

            curr_AUPR = get_AUPR(all_ys, all_predictions)

            #if  curr_AUPR > global_max:
            if  total_bce < global_max:
                #global_max = curr_AUPR
                global_max = total_bce
                best_epoch = epoch
                torch.save(classifier.state_dict(),
                           f"models/classifier_{direction}.pt")
            else:
                since_best += 1
                if since_best == 5:
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('CUDA', type=int)
    args = vars(parser.parse_args())
    CUDA = args["CUDA"]

    directions = [dir_.replace("_dir_dict.pkl", "")
                  for dir_ in os.listdir("../get_data/Shallow/directions/")[CUDA::8]]
    for direction in progressbar(directions):
        print(direction)
        save_trained(direction)


