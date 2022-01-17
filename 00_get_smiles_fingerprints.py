import pickle

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_smiles(path):
    with open(path, "r") as f:
        return set([line.split()[0] for line in f.readlines()])


def write_featurized(dataset, suffix):
    with open(f"../get_data/{dataset}/{dataset.lower()}_{suffix}", "r") as f:
        smiles = [line.split()[0] for line in f.readlines()]

    np.save(f"data/{dataset.lower()}_{suffix}_smiles_tokens.npy",
            FEATURIZER.featurize(smiles,
                                 log_every_n=1e5))


all_smiles = set()
for dataset in progressbar(["BindingDB", "DUDE"]):
    for suffix in ["actives", "decoys"]:
        all_smiles |= \
            get_smiles(f"../get_data/{dataset}/{dataset.lower()}_{suffix}")

all_smiles = list(all_smiles)
with open("data/all_smiles.pkl", "wb") as f:
    pickle.dump(all_smiles, f)

all_fingerprints = []
for smiles in progressbar(all_smiles):
    all_fingerprints.append(get_fingerprint(smiles))

print(all_fingerprints[0])

all_fingerprints = np.vstack(all_fingerprints)
np.save("data/all_fingerprints.npy", all_fingerprints)
print(all_fingerprints.dtype)
print(all_fingerprints.shape)

