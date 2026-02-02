
# %%
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
df = pd.read_parquet('../../data/processed.parquet')
# %%
df.columns
# %%
df = df.drop(columns='smiles')
df.columns

# %%
X = df.drop(columns='pic50')  # apenas covariaveis
y = df['pic50']  # pic50 é o target
# %%
print(X)
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape  # Compreendendo o conjunto de dados de treino

'''882 dados de treino, com 2063 colunas de explicativas
resultando num problema de dimensionalidade'''


# %%
X_test.shape, y_test.shape  # Compreendendo o conjunto de dados de teste

# %%
# Treinamento e Avaliação com Random Forest

rf = RandomForestRegressor(n_estimators=100,
                           random_state=42,
                           n_jobs=-1,
                           )
# Testando sem PCA: Random Forests lidam bem com alta dimensionalidade (Fingerprints)
# O PCA pode estar removendo padrões não-lineares importantes
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real')
plt.ylabel('Predito')
plt.title('Random Forest: Real vs Predito')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# %%
'''
Testando outros modelos de regressão.
XGBoost
Gradient Boosting
'''

modelo_xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_xgb.fit(X_train, y_train)
y_pred = modelo_xgb.predict(X_test)

print(f"XGBoost R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real')
plt.ylabel('Predito')
plt.title('XGBoost: Real vs Predito')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %%
