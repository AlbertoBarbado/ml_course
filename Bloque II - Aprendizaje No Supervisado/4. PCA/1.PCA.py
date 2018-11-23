# -*- coding: utf-8 -*-

###############################################################################

# PCA

###############################################################################


# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### Data Preprocessing
# Dataset
dataset = pd.read_csv('house_prices.csv', encoding='utf-8')

# Eliminar columnas con NaN
dataset_f = dataset[["MSSubClass", "LotFrontage", "LotArea",
             "GarageYrBlt", "GarageCars", "GarageArea", 
             "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
             "ScreenPorch", "PoolArea", "YrSold", "SalePrice"]].dropna()


dataset_f.describe()

f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(dataset_f.corr(method='pearson'),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()


X = dataset_f.iloc[:, 0:len(dataset_f.columns)-1].values
y = dataset_f.iloc[:, len(dataset_f.columns)-1].values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None) # 'None' para que conserven en principio todas las PC

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

"""
Se crea este vector para que  diga la varianza explicada por cada escenario de PCAs y 
ver que % de varianza explica cada componente.
Se Extraen todas las indep. variables porque se ha puesto 'None', pero se tiene un vector 
que ordena las que hay, el % que explican...
Se va sumando a medida que baja por el vector la varianza que explicaría si se cogen 2 variables, 3...

"""
explained_variance = pca.explained_variance_ratio_ 
print("Varianza Explicada por cada PC")
print(explained_variance)
var_exp = np.round(np.sum(explained_variance[0:5]),4)
print("Con 5 PC se explicaría el {var}% de la varianza".format(var=var_exp*100))
# Con los 5 ppales, se ve que el 65.3% de la varianza 

# Se entrena solo para esas 5 componentes principales
pca = PCA(n_components = 5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Varianza Explicada por cada PC")
print(explained_variance)
print("Parámetros del Modelo")
print(pca.components_)

# Visualizacion de las PC
sns.barplot(x='PC',y="var", 
           data=pd.DataFrame({'var':explained_variance,
             'PC':['PC1','PC2','PC3','PC4', 'PC5']}), color="c")

### Modelo de Regresión
# Con las PCA se construye un modelo de regresión

# Regresion Lineal
import statsmodels.api as sm
model = sm.OLS(y_train, X_train_pca).fit()
model.summary() # Se ve que la PC realmente relevante es solo la primera

# RF
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=5, random_state=0,
                               n_estimators=100)
model.fit(X_train_pca, y_train)
print("Relevancia de los parámetros")
print(model.feature_importances_) # Aparentemente con la primera componente artificial construida es suficiente

# Predicciones
y_pred = model.predict(X_test_pca)

# Metricas de evaluacion
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)


# Usando solo 1 PC
pca = PCA(n_components = 1)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Varianza Explicada por cada PC")
print(explained_variance)
print("Parámetros del Modelo")
print(pca.components_)

model = RandomForestRegressor(max_depth=5, random_state=0,
                               n_estimators=100)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae) # Mejoran, de hecho, los resultados

# Usando 2 PC para visualizar
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Varianza Explicada por cada PC")
print(explained_variance)
print("Parámetros del Modelo")
print(pca.components_)

model = RandomForestRegressor(max_depth=5, random_state=0,
                               n_estimators=100)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)

plt.scatter(X_train_pca[:,0], X_train_pca[:,1])
plt.ylabel("PC1")
plt.xlabel("PC2")
plt.title("Representación Gráfica de las PC")
plt.show()



