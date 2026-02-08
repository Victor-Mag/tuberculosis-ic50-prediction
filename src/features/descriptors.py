# %%
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pyarrow.parquet as pq
import pyarrow as pa
from morgan_descriptors_generator import descritores
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('../../data/raw.csv', sep=',')
df.head()
# %%
df.columns
# %%
# Removendo Unnamed:0
df = df.drop(columns=['Unnamed: 0', 'ic50_nm'])
df.head()
# %%
features_lista = []
smiles_validos = []
pic50_validos = []

# %%

for idx, row in df.iterrows():
    smiles = row['canonical_smiles']
    pic50 = row['pic50']

    features = descritores(smiles)

    if features is not None:
        features_lista.append(features)
        smiles_validos.append(smiles)
        pic50_validos.append(pic50)
    else:
        print(f" Smiles invalido na linha: {idx}: {smiles}")

X = np.array(features_lista)
y = np.array(pic50_validos)

# %%
print(f"\n{'='*60}")
print("RESULTADO:")
print(f"{'='*60}")
print(f"Moléculas processadas: {len(features_lista)}/{len(df)}")
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"Features totais: {X.shape[1]} Fingerprints de Morgan)")

print(f"\nVerificação de dados:")
print(f"  NaN em X: {np.isnan(X).sum()}")
print(f"  Inf em X: {np.isinf(X).sum()}")
print(f"  NaN em y: {np.isnan(y).sum()}")
# %%
print(f"\nEstatísticas dos fingerprints:")
print(f"Features não nulas por molécula (média): {X.sum(axis=1).mean():.1f}")
print(f"Densidade de Features: {X.mean():.3f}")  # Fraction of bits that are 1
print(f"Uso de memória: {(X.nbytes + y.nbytes) / 1024**2:.1f} MB")


# %%
'''
Salvando os descritores de morgan processados num parquet
'''

df_processado = pd.DataFrame(X)
df_processado['pic50'] = y
df_processado['smiles'] = smiles_validos

# %%
df_processado.to_parquet('../../data/processed.parquet', index=False)

# %%
'''
Salvando todos os descritores físico-quimicos (1D) em outro parquet
para comparar o modelo usando fingerprints de morgan x descritores físico-quimicos
'''

df_desc = df2.copy()
df_desc.head()

# %%
nomes_propriedades = list(rdMolDescriptors.Properties.GetAvailableProperties())
get_propriedades = rdMolDescriptors.Properties(nomes_propriedades)

# %%

tqdm.pandas()
# %%


def smiles_para_prop(smiles):
    mol = Chem.MolFromSmiles(smiles)
    props = None
    if mol:
        mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#1x0]'))
        props = np.array(get_propriedades.ComputeProperties(mol))
    return props


# %%
df_desc['props'] = df_desc['canonical_smiles'].progress_apply(smiles_para_prop)
df_desc[nomes_propriedades] = df_desc['props'].to_list()

# %%
df_desc.head()
# %%
df_desc = df_desc.drop(columns=['props'])
# %%
df_desc.head()



# %%
'''
Aplicando scaler em outro dataframe desses descritores para verificar qual tem melhor desempenho
'''
df_desc.columns


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_desc_scaled = scaler.fit_transform(df_desc[['exactmw', 'amw', 'lipinskiHBA',
       'lipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumHeavyAtoms',
       'NumAtoms', 'NumHeteroatoms', 'NumAmideBonds', 'FractionCSP3',
       'NumRings', 'NumAromaticRings', 'NumAliphaticRings',
       'NumSaturatedRings', 'NumHeterocycles', 'NumAromaticHeterocycles',
       'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumSpiroAtoms',
       'NumBridgeheadAtoms', 'NumAtomStereoCenters',
       'NumUnspecifiedAtomStereoCenters', 'labuteASA', 'tpsa', 'CrippenClogP',
       'CrippenMR', 'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v', 'chi0n',
       'chi1n', 'chi2n', 'chi3n', 'chi4n', 'hallKierAlpha', 'kappa1', 'kappa2',
       'kappa3', 'Phi']])

# %%
print(df_desc_scaled)

# %%
rows, columns = df_desc_scaled.shape
itens_por_linha = columns

print(f"Existem cerca de {itens_por_linha} itens em {rows} colunas")

# %%

cols = ['exactmw', 'amw', 'lipinskiHBA',
       'lipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumHeavyAtoms',
       'NumAtoms', 'NumHeteroatoms', 'NumAmideBonds', 'FractionCSP3',
       'NumRings', 'NumAromaticRings', 'NumAliphaticRings',
       'NumSaturatedRings', 'NumHeterocycles', 'NumAromaticHeterocycles',
       'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumSpiroAtoms',
       'NumBridgeheadAtoms', 'NumAtomStereoCenters',
       'NumUnspecifiedAtomStereoCenters', 'labuteASA', 'tpsa', 'CrippenClogP',
       'CrippenMR', 'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v', 'chi0n',
       'chi1n', 'chi2n', 'chi3n', 'chi4n', 'hallKierAlpha', 'kappa1', 'kappa2',
       'kappa3', 'Phi']

for i in df_desc_scaled:
    for j in i:
        df_scaled = pd.DataFrame(df_desc_scaled, columns = cols)
# %%
df_scaled.head()

# %%
df_scaled_comb = pd.concat([df_scaled, df_desc[['canonical_smiles', 'pic50']]], axis=1)
df_scaled_comb.columns

# %%
df_scaled_comb.to_parquet('../../data/processed_scaled_desc.parquet')
df_desc.to_parquet('../../data/processed_desc.parquet')