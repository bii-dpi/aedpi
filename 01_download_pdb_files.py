import os

import pandas as pd

from Bio.PDB import *
from progressbar import progressbar


pdbl = PDBList(verbose=False)

DATASETS = ["BindingDB", "DUDE"]


def download_pdbs(dataset):
    with open(f"../get_data/{dataset}/{dataset.lower()}_actives", "r") as f:
        pdb_ids = set([line.split()[1].split("_")[0]
                       for line in f.readlines()])

    pdbl.download_pdb_files(pdb_ids, file_format="pdb", pdir="data/pdb_files")

    for pdb_id in pdb_ids:
        os.rename(f"data/pdb_files/pdb{pdb_id.lower()}.ent",
                  f"data/pdb_files/pdb{pdb_id.lower()}.pdb")


for dataset in progressbar(DATASETS):
    download_pdbs(dataset)

