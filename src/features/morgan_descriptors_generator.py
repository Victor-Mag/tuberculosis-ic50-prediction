# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors, GraphDescriptors, MolFromSmiles
from rdkit import DataStructs


def descritores(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)

    desc = np.array([
        Crippen.MolLogP(mol),              # Lipofilicidade
        Descriptors.TPSA(mol),             # Permeabilidade
        Descriptors.MolWt(mol),            # Peso
        Lipinski.NumHDonors(mol),          # H-bond
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),   # Flexibilidade
        Lipinski.NumAromaticRings(mol),    # Aromaticidade
        Lipinski.FractionCSP3(mol),        # Saturação
        Lipinski.HeavyAtomCount(mol),      # Tamanho
        Lipinski.RingCount(mol),           # Ciclicidade
        Lipinski.NumHeteroatoms(mol),
        Crippen.MolMR(mol),                # Refração
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        GraphDescriptors.BertzCT(mol),     # Complexidade
    ])

    return np.concatenate([fp_array, desc])
