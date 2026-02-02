'''
Docstring for features.preprocessing
O intuito desse arquivo é aplicar a normalização e redução de dimensionalidade
usando PCA e t-sne

Tem um caráter investigativo também, uma vez que as covariáveis são tanto
binárias (fingerprints de Morgan) quanto contínuas (descritores)

'''
# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_parquet('../../data/processed.parquet')
# %%
df.head()
# %%
X = df.drop(columns=['pic50','smiles'])
print(X)
# %%
X_scaled = StandardScaler().fit_transform(X)
# %%
print(X_scaled)

# %%
import numpy as np
from sklearn.decomposition import PCA



# Salvando as colunas e indices originais para aplicar posteriormente no X_train
# Aplicando PCA
indices_originais = X_train.index
colunas_originais = X_train.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# %%
X_train = pd.DataFrame(X_train, columns=colunas_originais,
                       index=indices_originais)
# %%
print(X_train)
# %%
X_train.describe().round(3)

# %%
pca = PCA()
X_pca = pca.fit_transform(X_train)

# %%
pca.explained_variance_ratio_

# %%
# %%
plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red',
         label='Cumulative Explained Variance')
plt.legend(loc='upper left')
ax = plt.gca()
ax.set_xlim([0, 400])
ax.set_ylim([0, 1.1])
# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Componentes')
plt.ylabel('Variância Acumulada')
plt.show()

# %%
'''
É possível perceber que 250 componentes respondem a cerca de 80 por cento
da variância. Logo é possível reduzir a dimensionalidade do meu modelo para
em torno de 260 componentes sem perder capacidade de generalização.

Entretanto, para Random Forest com Fingerprints, o PCA causou perda de performance.
Optamos por usar os dados originais (apenas escalados).
'''
# %%

indices_originais_test = X_test.index
colunas_originais_test = X_test.columns

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, index=indices_originais_test,
                      columns=colunas_originais_test)


