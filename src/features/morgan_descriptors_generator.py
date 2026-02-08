# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit import DataStructs


def descritores(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.array(fp)

    return fp_array
