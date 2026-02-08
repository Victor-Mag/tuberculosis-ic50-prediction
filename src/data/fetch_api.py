"""
Script para extrair dados de MIC para Mycobacterium tuberculosis do ChEMBL
Autor: Victor Hugo Magalhaes
Data: 2026
"""
# %%
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# %%
molecule = new_client.molecule
activity = new_client.activity

# %%
# Buscar atividades de LD50
print("Buscando dados de IC50...")
ic50_activities = activity.filter(
    target_chembl_id='CHEMBL360',
    standard_type='IC50',
    # standard_units=['ug.mL-1'],
    standard_relation='=',
    assay_type__in=['F', 'B']  # Funcionais e padrões biologicos
).only([
    'molecule_chembl_id',
    'canonical_smiles',
    'standard_value',
    'standard_units',
    'assay_description',
    'assay_type',
    'target_organism'
])

# %%
# Convertendo para lista

ic50_list = list(ic50_activities)[:7000]

print(f"Registros encontrados: {len(ic50_list)}")

# %%

df = pd.DataFrame(ic50_list)
df.tail()

# %%


def converter_para_nM(row):
    """
    Converte diferentes unidades de IC50 para nanomolar (nM)
    """
    valor = row['standard_value']
    unidade = row['standard_units']
    smiles = row.get('canonical_smiles')

    if pd.isna(valor) or pd.isna(unidade):
        return None

    # Dicionário de fatores de conversão para nM
    conversao = {
        'nM': 1,
        'uM': 1000,          # micromolar → nanomolar
        'µM': 1000,
        'mM': 1000000,     # milimolar → nanomolar
        'M': 1000000000,  # molar → nanomolar
        'pM': 0.001,         # picomolar → nanomolar
        'fM': 0.000001,      # femtomolar → nanomolar
    }

    # Unidades de massa/volume (requer peso molecular - aproximação)
    # Fatores para converter para g/L
    fator_para_g_L = {
        'ug.mL-1': 1e-3,     # ug/mL = mg/L = 10^-3 g/L
        'ug/ml': 1e-3,
        'ng.mL-1': 1e-6,     # ng/mL = 10^-6 g/L
        'ng/ml': 1e-6,
        'mg.mL-1': 1,        # mg/mL = g/L
        'mg/ml': 1,
    }

    # Normalizar a unidade
    unidade_clean = unidade.strip()

    if unidade_clean in conversao:
        return valor * conversao[unidade_clean]
    elif unidade_clean in fator_para_g_L:
        # Calcula MW exato pelo SMILES
        if smiles and pd.notna(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    concentracao_g_L = valor * fator_para_g_L[unidade_clean]
                    molaridade = concentracao_g_L / mw
                    return molaridade * 1e9  # Retorna em nM
            except:
                pass
        return None
    else:
        # print(f"Unidade desconhecida: {unidade}")
        return None


# %%
df['standard_value'].dtype
# %%
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
print(df['standard_value'].dtype)

# %%
df['ic50_nm'] = df.apply(converter_para_nM, axis=1)
df['pic50'] = -np.log10(df['ic50_nm']*1e-9)

# %%
# Filtragem de valores biologicamente implausíveis
print(f"Removendo {(df['pic50'] < 0).sum()} registros com pIC50 negativo (erro de unidade/artefato)...")
df = df[df['pic50'] >= 0].copy()


# %%
df.head(5)
# %%
print(f"Máximo: {df['pic50'].max()}")
print(f"Mínimo: {df['pic50'].min()}")
# %%
df['molecule_chembl_id'].nunique()
df = df.drop_duplicates(subset=['molecule_chembl_id'])

# %%
df.info()
# %%
df['pic50'].unique()

# %%
plt.figure(figsize=(10, 6))
plt.hist(df['pic50'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribuição de pIC50')
plt.xlabel('pIC50')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.5)
plt.show()

# %%
'''
Estatísticas Básicas
'''

print('='*50)
print('Estatísticas pIC50')
print('='*50)
print(f"Total de amostras", len(df['pic50']))
print(f"\n{df['pic50'].describe():}\n")
print(f"Assimetria: {df['pic50'].skew():.2f}")
print(f"Curtose: {df['pic50'].kurt():.2f}")
# %%
minimo = df['ic50_nm'].min()
df.loc[df['ic50_nm'] == minimo]
# %%
df.columns

# %%
df_atualizado = df.drop(columns=['assay_description',
                                'assay_type',
                                'molecule_chembl_id',
                                'standard_units',
                                'standard_value',
                                'target_organism',
                                'type',
                                'units',
                                'value'])

# %%
df_atualizado.to_csv('../../data/raw.csv', sep=',')
