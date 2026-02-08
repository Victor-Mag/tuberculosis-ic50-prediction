
# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

df = pd.read_parquet('../../data/processed.parquet')
# %%
df.columns, df.shape
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
# Criando bins (categorias) do target para permitir a estratificação
# Isso garante que treino e teste tenham distribuições de pic50 similares
bins = pd.cut(y, bins=8, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=bins)

# Compreendendo o conjunto de dados de treino
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%

lista_modelos = {
    'RandomForest': RandomForestRegressor(n_estimators=100,
                                          random_state=42,
                                          n_jobs=-1,
                                          max_depth=6),
    'XGBoost': XGBRegressor(n_estimators=100,
                            random_state=42,
                            n_jobs=-1,
                            max_depth=6,
                            eta=0.7,
                            gamma=20),
    'SVR': SVR(kernel='rbf',
               gamma='auto',
               C=0.1)
}

for nome, modelo in lista_modelos.items():
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=modelo,
        X=X,
        y=y,
        train_sizes=np.linspace(0.2, 1, 4),
        n_jobs=-1,
        scoring='r2',
        cv=10,
        verbose=1
    )
    print(f'Nome: {nome}, r2: {val_scores.mean():.3f} ± {val_scores.std():.3f}')

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label='Training R²')
    plt.plot(train_sizes, val_mean, label='Validation R²')

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2
    )

    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2
    )

    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend()
    plt.show()


# %%
# Treinamento e Avaliação com Random Forest

rf = RandomForestRegressor(n_estimators=100,
                           random_state=42,
                           n_jobs=-1,
                           criterion='squared_error')


# %%
train_sizes, train_scores, val_scores = learning_curve(estimator=rf,
                                                       X=X,
                                                       y=y,
                                                       train_sizes=np.linspace(
                                                           0.1, 1.0, 6),
                                                       cv=5,
                                                       scoring='r2',
                                                       n_jobs=-1)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label='Training R²')
plt.plot(train_sizes, val_mean, label='Validation R²')

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.2
)

plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.show()


# %%
# Calculando Q^2 (R2 de validação cruzada)
rf = RandomForestRegressor(n_estimators=100,
                           random_state=42,
                           n_jobs=-1,
                           criterion='absolute_error')

scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
q2 = scores.mean()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Q² (CV): {q2:.3f} (+/- {scores.std():.3f})")
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

modelo_xgb = XGBRegressor(n_estimators=100,
                           random_state=42,
                            n_jobs=-1)

# Calculando Q^2 para XGBoost
scores_xgb = cross_val_score(modelo_xgb, X_train, y_train, cv=5, scoring='r2')
q2_xgb = scores_xgb.mean()

modelo_xgb.fit(X_train, y_train)
y_pred = modelo_xgb.predict(X_test)

print(f"XGBoost Q² (CV): {q2_xgb:.3f} (+/- {scores_xgb.std():.3f})")
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
'''
XGBOOST:       R² = 0.652, RMSE = 0.611
Random Forest: R² = 0.658, RMSE = 0.606
'''


# %%
'''
Teste com descritores daqui pra baixo
'''
# %%
lista_var = []
df_desc = pd.read_parquet('../../data/processed_desc.parquet')
df_desc_scaled = pd.read_parquet('../../data/processed_scaled_desc.parquet')

# %%
df_desc.head(), df_desc_scaled.head()

# %%
df_desc.columns

# %%
X_desc = df_desc.drop(columns=['canonical_smiles', 'pic50'])
X_scaled = df_desc.drop(columns=['canonical_smiles', 'pic50'])

X_scaled.columns, X_desc.columns
# %%
lista_var = [X_desc, X_scaled]

# %%
y = df_desc['pic50']
# %%

for i in lista_var:
    X_train, X_test, y_train, y_test = train_test_split(
        i, y, test_size=0.2, random_state=42)

    rf_model_desc = RandomForestRegressor()
    rf_model_desc.fit(X_train, y_train)
    y_pred = rf_model_desc.predict(X_test)

    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.title('Random Forest: Real vs Predito')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

'''
É possível perceber que os descritores não obtiveram um bom rendimento
os Fingerprints de Morgan são mais promissores
Apenas os descritores: R² = 0.52 e RMSE = 0.718
Descritores c/ scale: R² = 0.535 e RMSE = 0.707
'''

# %%
