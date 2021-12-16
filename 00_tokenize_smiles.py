import numpy as np
import deepchem as dc

from rdkit import Chem
from progressbar import progressbar


FEATURIZER = dc.feat.Mol2VecFingerprint()

DATASETS = ["BindingDB", "DUDE"]


def write_featurized(dataset, suffix):
    with open(f"../get_data/{dataset}/{dataset.lower()}_{suffix}", "r") as f:
        smiles = [line.split()[0] for line in f.readlines()]

    np.save(f"data/{dataset.lower()}_{suffix}_smiles_tokens.npy",
            FEATURIZER.featurize(smiles,
                                 log_every_n=1e5))


for dataset in progressbar(DATASETS):
    for suffix in ["actives", "decoys"]:
        write_featurized(dataset, suffix)

