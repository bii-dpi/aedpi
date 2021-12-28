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
from concurrent.futures import ProcessPoolExecutor


POCKET_CUTOFF = 10.
RESOLUTION = 1

elements_list = ['C', 'N', 'O', 'P', 'S', 'F', 'CL', 'BR', 'I']
elements_dict = dict(zip(elements_list, range(1, len(elements_list) + 1)))

p = PDBParser(PERMISSIVE=1)


def get_pocket_box(fname):
    coords, _ = load_molecule(f"data/pdb_files/{fname}",
                              is_protein=True,
                              sanitize=False)
    # XXX: need to make sure the other options for load_molecule are also okay
    # for loading proteins for pocket-finding purposes.
    return get_face_boxes(coords, 5.)[0]


def get_pocket_atoms(structure, pocket_box):
    # Append the protein's residues to a list.
    protein_residues = []
    for residue in structure.get_residues():
        is_hetero = residue.get_id()[0].startswith("H_")
        if not is_hetero and is_aa(residue):
            protein_residues.append(residue)

    # Append all of the atoms across these residues into a list.
    protein_atoms = [
        atom for residue in protein_residues for atom in residue.get_unpacked_list()
    ]

    # Return a matrix of their coordinates, and a list of corresponding
    # elements.
    protein_elements = [atom.element for atom in protein_atoms]
    protein_coords = [atom.get_coord() for atom in protein_atoms]

    pocket_indices = [pocket_box.__contains__(coords)
                      for coords in protein_coords]

    pocket_coords = np.array(protein_coords)[pocket_indices]
    pocket_elements = np.array(protein_elements)[pocket_indices]

    return pocket_coords, pocket_elements


def center_coords(pocket_box, pocket_coords):
    pocket_coords[:, 0] -= pocket_box.x_range[0]
    pocket_coords[:, 1] -= pocket_box.y_range[0]
    pocket_coords[:, 2] -= pocket_box.z_range[0]

    return pocket_coords


def encode_pocket_elements(pocket_elements):
    encoded = []
    for pocket_element in pocket_elements:
        try:
           encoded.append(elements_dict[pocket_element])
        except:
           encoded.append(10)

    return encoded


def get_pocket_grid(pocket_coords, pocket_elements):
    def get_range(col):
        values = pocket_coords[:, col].flatten()
        return round(np.max(values))

    pocket_coords = np.rint(pocket_coords).astype(np.int16)
    grid = np.zeros([get_range(col) for col in range(3)])

    for i in range(len(pocket_coords)):
        x, y, z = pocket_coords[i, :] - 1
        grid[x][y][z] = max(grid[x][y][z], pocket_elements[i])

    return grid



def write_protein_pocket_grid(fname):
    pdb_id = fname[3:-4].upper()
    structure = p.get_structure(pdb_id, f"data/pdb_files/{fname}")

    pocket_box = get_pocket_box(fname)
    pocket_coords, pocket_elements = get_pocket_atoms(structure, pocket_box)

    pocket_coords = center_coords(pocket_box, pocket_coords)
    pocket_elements = encode_pocket_elements(pocket_elements)

    np.save(f"data/grids/{pdb_id}_grid",
            get_pocket_grid(pocket_coords, pocket_elements))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        for fname in progressbar(os.listdir("data/pdb_files")):
            pdb_id = fname[3:-4].upper()
            if not os.path.isfile(f"data/grids/{pdb_id}_grid.npy"):
                print(f"data/grids/{pdb_id}_grid")
                write_protein_pocket_grid(fname)
        """
        with ProcessPoolExecutor() as executor:
            executor.map(write_protein_pocket_grid,
                         os.listdir("data/pdb_files"))
        """

