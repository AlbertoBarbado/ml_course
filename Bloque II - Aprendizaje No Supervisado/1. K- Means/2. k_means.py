# -*- coding: utf-8 -*-

###############################################################################

# K-Means Clustering

###############################################################################


# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('CC GENERAL.csv', encoding='utf-8')
X = dataset[["BALANCE", "PURCHASES"]].values

# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init= 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X) 

### Visualizar clusters (se pintan uno a uno)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'C1') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'C2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'C5')

# Para pintar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroides') 
plt.title('Clusters de Clientes')
plt.xlabel('X1: Balance en la Cuenta ($)')
plt.ylabel('X2: Gastos en Compras ($)')
plt.legend()
plt.show()