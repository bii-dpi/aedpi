import torch
from model import Classifier
from dataloader import get_dataloaders
from progressbar import progressbar
import torch.nn.functional as F


DIRECTION = "bztdz"
CUDA = 0
BATCH_SIZE = 128
SEED = 12345
LR = 1e-4
LAMBDA = 1e-2


device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
training_dl, validation_dl = get_dataloaders(DIRECTION, True, SEED,
                                             BATCH_SIZE)

classifier = Classifier().to(device)
#model.load_state_dict(torch.load(f"models/classifier_{DIRECTION}.pt", map_location="cpu"))
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

def get_loss(decoded_proteins, proteins, decoded_ligands, ligands, predictions, y):
    BCE = F.binary_cross_entropy(predictions.flatten(), y.float(), size_average=False)
    MSE_protein = F.mse_loss(decoded_proteins, proteins, size_average=False)
    # Need MSE or BCE for ligands?
    MSE_ligand = F.mse_loss(decoded_ligands, ligands, size_average=False)

    return BCE + LAMBDA * (MSE_protein + MSE_ligand), BCE, MSE_protein + MSE_ligand


rows = ["tr_both,tr_bce,tr_mse,val_both,val_bce,val_mse"]
epochs = 100
global_min, best_epoch = 1e6, -1
for epoch in range(epochs):
    total_both, total_bce, total_mse = 0, 0, 0
    for proteins, ligands, y in training_dl:
        optimizer.zero_grad()
        proteins, ligands, y = (proteins.to(device),
                                ligands.to(device),
                                y.to(device))
        decoded_proteins, decoded_ligands, predictions = \
            classifier(proteins, ligands)

        both, bce, mse = get_loss(decoded_proteins, proteins,
                                  decoded_ligands, ligands,
                                  predictions, y)

        total_both += both.data.item()
        total_bce += bce.data.item()
        total_mse += mse.data.item()
        both.backward()
        optimizer.step()

    print(f"Epoch[{epoch + 1}/{epochs}] Loss: {total_both / len(training_dl):.3f}")
    curr_row = f"{total_both},{total_bce},{total_mse},"

    if epoch % 1 == 0:
        with torch.no_grad():
            total_both, total_bce, total_mse = 0, 0, 0
            for proteins, ligands, y in validation_dl:
                proteins, ligands, y = (proteins.to(device),
                                        ligands.to(device),
                                        y.to(device))
                decoded_proteins, decoded_ligands, predictions = \
                    classifier(proteins, ligands)

                both, bce, mse = get_loss(decoded_proteins, proteins,
                                          decoded_ligands, ligands,
                                          predictions, y)

                total_both += both.data.item()
                total_bce += bce.data.item()
                total_mse += mse.data.item()
            curr_row += f"{total_both},{total_bce},{total_mse}"

        rows.append(curr_row)

        if  total_bce < global_min:
            global_min = total_bce
            best_epoch = epoch
            torch.save(classifier.state_dict(),
                       f"models/classifier_{DIRECTION}.pt")

        print(global_min, best_epoch)


with open("results.csv", "w") as f:
    f.write("\n".join(rows))

