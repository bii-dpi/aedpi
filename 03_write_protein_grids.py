import os
import warnings

import numpy as np

from Bio.PDB import *
from progressbar import progressbar
from deepchem.utils.coordinate_box_utils \
  import CoordinateBox, get_face_boxes
from deepchem.utils.rdkit_utils import load_molecule
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from pdbfixer import PDBFixer




POCKET_CUTOFF = 10.

p = PDBParser(PERMISSIVE=1)


def get_pocket_box(fname):
    coords, _ = load_molecule(f"data/pdb_files/{fname}", is_protein=True)
    return get_face_boxes(coords, 5.)[0]


def get_pocket_grid(structure, pocket_box):
    # Append the ligand's residues to a list.
    protein_residues = []
    for residue in structure.get_residues():
        is_hetero = residue.get_id()[0].startswith("H_")
        if not is_hetero and is_aa(residue):
            protein_residues.append(residue)

    # Append all of the atoms across these residues into a list.
    protein_atoms = [
        atom for residue in protein_residues for atom in residue.get_unpacked_list()
    ]

    # Return a matrix of their coordinates.
    pocket_coords = [atom.get_coord() for atom in protein_atoms]
    print(np.array(pocket_coords).shape)
    pocket_coords = np.array([coords for coords in pocket_coords
                              if pocket_box.__contains__(coords)])
    print(pocket_coords.shape)
    return pocket_coords


def write_protein_pocket_grid(fname):
    pdb_id = fname[3:-4].upper()
    structure = p.get_structure(pdb_id, f"data/pdb_files/{fname}")

    pocket_box = get_pocket_box(fname)

    np.save(f"data/grids/{pdb_id}_grid",
            get_pocket_grid(structure, pocket_box))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        for fname in progressbar(os.listdir("data/pdb_files")):
            write_protein_pocket_grid(fname)

