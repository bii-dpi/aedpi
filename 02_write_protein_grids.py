import os
import warnings

import numpy as np

from Bio.PDB import *
from progressbar import progressbar
from Bio.PDB.PDBExceptions import PDBConstructionWarning


POCKET_CUTOFF = 10.

p = PDBParser(PERMISSIVE=1)


def get_ligand_centroid(structure):
    """Return a matrix of the ligand's atom coordinates."""
    # Append the ligand's residues to a list.
    ligand_residues = []
    for residue in structure.get_residues():
        is_hetero = residue.get_id()[0].startswith("H_")
        if is_hetero and not is_aa(residue):
            ligand_residues.append(residue)

    # Append all of the atoms across these residues into a list.
    ligand_atoms = [
        atom for residue in ligand_residues for atom in residue.get_unpacked_list()
    ]

    # Return the centroid of their coordinates.
    return np.array([atom.get_coord() for atom in ligand_atoms]).mean(0)


def get_pocket_grid(structure, ligand_centroid):
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
    pocket_coords = np.array([atom.get_coord() for atom in protein_atoms])
    distances = np.linalg.norm(pocket_coords - ligand_centroid,
                               ord=2, axis=1)
    pocket_coords = pocket_coords[distances <= POCKET_CUTOFF]
    return pocket_coords


def write_protein_pocket_grid(fname):
    pdb_id = fname[3:-4].upper()
    structure = p.get_structure(pdb_id, f"data/pdb_files/{fname}")
    ligand_centroid = get_ligand_centroid(structure)
    """
    np.save(f"data/grids/{pdb_id}_grid",
            get_pocket_grid(structure, ligand_centroid))
    """


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        for pdb_id in progressbar(os.listdir("data/pdb_files")):
            write_protein_pocket_grid(pdb_id)

