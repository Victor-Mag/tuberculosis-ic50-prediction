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
    standard_units='nM',
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
df_ic50 = df['assay_description']
df_ic50.value_counts().head(5)  # Isso confirma que os dados tem natureza
# Fenotipica, logo tratam apenas do Mycobacterium como um todo
# Não sobre enzimas ou vias metabólicas suprimidas

# %%
df.to_csv('../../data/raw.csv', sep=',')
