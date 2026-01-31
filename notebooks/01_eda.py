# %%
import pandas as pd
# %%
df_raw = pd.read_csv("../data/raw.csv", sep=",")
df_raw.head()

# %% 
print(f"Tipos de valores \n: {df_raw.dtypes}")
#%%
print(f"\nshape do df: {df_raw.shape}")
#%%
print(f"Estatisticas: {df_raw.describe()}")
#%%
df_raw.isnull().sum()

# %%
df_raw.nunique()

# %%
# Deixando apenas as as linhas com valores unicos de canonical smiles
df_un = df_raw.drop_duplicates(['canonical_smiles'])
# %%
df_un.shape

#%% 
df_un['value'].head(10)

#%%
print(df_un['standard_value'].max())
print(df_un['standard_value'].min())
print(df_un['standard_value'].describe())
# %%

df_un['standard_value'].isna().sum()

# Todos valores estão preenchidos =)

# %%
selecao = ['molecule_chembl_id','canonical_smiles', 'standard_value']
df3 = df_un[selecao]
df3
#%%
print(f"Assimetria: {df3['standard_value'].skew()}")
print(f"Curtose: {df3['standard_value'].kurt()}")

#%%
import matplotlib.pyplot as plt
df3['standard_value'].plot.kde()
#plt.xscale('log')
plt.semilogx()
plt.show()

# %%
import numpy as np


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribuição original
axes[0].hist(df3['standard_value'].dropna(), bins=50)
axes[0].set_xlabel('IC50 (escala linear)')
axes[0].set_title('Original - MUITO assimétrico!')

# Log scale
axes[1].hist(np.log10(df3['standard_value'].dropna()), bins=50)
axes[1].set_xlabel('log10(IC50)')
axes[1].set_title('Escala log10')

# Boxplot para ver outliers
axes[2].boxplot(np.log10(df3['standard_value'].dropna()))
axes[2].set_ylabel('log10(IC50)')
axes[2].set_title('Outliers?')

plt.tight_layout()
plt.show()

#%%

print("Top 10 valores mais altos:")
print(df_un.nlargest(10, 'standard_value')[['canonical_smiles', 'standard_value', 'standard_units']])

print(f"\nValores > 1M: {(df_un['standard_value'] > 1e6).sum()}")
print(f"Valores > 1 bilhão: {(df_un['standard_value'] > 1e9).sum()}")

#%%
df3['pic50'] = -np.log10(df3['standard_value'] * 1e-9)
#%%
df3.head()
# %%

plt.hist(df3['pIC50'].dropna(), bins=50)
plt.xlabel('pIC50')
plt.show()
# %%

print("Distribuição de atividade:")
print(f"Muito potente (pIC50 > 7): {(df3['pic50'] > 7).sum()}")
print(f"Potente (6 < pIC50 ≤ 7): {(df3['pic50'].between(6, 7)).sum()}")
print(f"Moderado (5 < pIC50 ≤ 6): {(df3['pic50'].between(5, 6)).sum()}")
print(f"Fraco (4 < pIC50 ≤ 5): {(df3['pic50'].between(4, 5)).sum()}")
print(f"Inativo (pIC50 ≤ 4): {(df3['pic50'] <= 4).sum()}")

#%%

df_clean = df3[df3['pic50'].between(3, 9)].copy()
print(f"Antes: {len(df3)} moléculas")
print(f"Depois: {len(df_clean)} moléculas")
print(f"Removidos: {len(df3) - len(df_clean)} ({100*(len(df3)-len(df_clean))/len(df3):.1f}%)")

#%%
df_clean['pic50'].plot.hist()
plt.semilogx()
plt.show()

# %%
print(f"Tamanho do dataset: {len(df_clean)} itens")
print(f"Média: {df_clean['pic50'].mean():.2f}")
print(f"Mediana: {df_clean['pic50'].median():.2f}")
print(f"Desvio Padrão: {df_clean['pic50'].std():.2f}")
print(f"Simetria: {df_clean['pic50'].skew()}")
print(f"Curtose: {df_clean['pic50'].kurt()}")


'''
Outro método de limpeza, mas não foi utilizado
Q1 = df3['pIC50'].quantile(0.25)
Q3 = df3['pIC50'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df['pIC50'] >= Q1 - 3*IQR) & 
    (df['pIC50'] <= Q3 + 3*IQR)
]
'''
# %%
bins = [0, 4, 5, 6, 7, 10]

labels = ['Fraco(<4)',
           'Moderado(4-5)',
            'Bom(5-6)',
            'Potente(6-7)',
            'Muito Potente(>7)']

df_clean['faixa'] = pd.cut(df_clean['pic50'], bins=bins, labels=labels)

print("\nDistribuição por faixa de atividade:")
print(df_clean['faixa'].value_counts().sort_index())

print("\nPercentual:")
print(df_clean['faixa'].value_counts(normalize=True).sort_index() * 100)

# %%
'''
Compreender quais descritores utilizar para o modelo =)
Fingerprints de morgan, descritores de lipinski e outros descritores fisico-quimicos

-> Possivelmente a estratégia campeã será Fingerprints + Descritores de TB
'''

# %%
from pathlib import Path
filepath = Path('../data/processed.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(filepath)
