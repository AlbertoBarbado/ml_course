# -*- coding: utf-8 -*-

###############################################################################

# Clustering Jerárquico

###############################################################################

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('CC GENERAL.csv', encoding='utf-8')
X = dataset[["BALANCE", "PURCHASES"]].values

### Dendograma para tener el numero optimo de clusters
# Se va a usar a usar una librería nueva, y con ello busco ver el numero optimo de clusters
import scipy.cluster.hierarchy as sch

# Se usa el metodo 'ward' que intenta minimizar la varianza entre clusters. 
# En lugar de minimizar el WC minimal square, se hace con la varianza -> minimizar la varianza en cada cluster
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) 
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancias Euclideas')
plt.show()

# Entrenar el algoritmo con los datos del conjunto
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X) # Con fit_predict se obtiene el cluster asignado a cada punto

### Visualizar clusters (se pintan uno a uno)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'blue', label = 'C1') 
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red', label = 'C2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'C3')

# Para pintar los centroides
plt.title('Clusters de Clientes')
plt.xlabel('X1: Balance en la Cuenta ($)')
plt.ylabel('X2: Gastos en Compras ($)')
plt.legend()
plt.show()