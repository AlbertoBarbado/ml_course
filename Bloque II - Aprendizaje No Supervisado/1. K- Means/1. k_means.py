# -*- coding: utf-8 -*-

###############################################################################

# K-Means Clustering

###############################################################################


# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('movies.csv', encoding='latin-1')
X = dataset[["budget", "gross"]].values

# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init= 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X) # fit_predict devuelve para cada punto a que cluster pertenece

### Visualizar clusters (se pintan uno a uno)
# Con esto lo que se hace es especificar que se quiere aplicar esto a los puntos del Cluster 1 (index=0), 
# y para la columna 0 (la de los valores de X) y la 1 (la de los valores de y) de esos puntos
# Es decir, en X se tienen dos columnas, se coge la de X (x[0]) y la de y (X[1]) pero aplicado a los puntos de cada cluster 
# (es decir, en vez de todas las filas con [:,] cojo solo las de un cluster [y_label == i])
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'C1') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'C2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'C4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'C5')

# Para pintar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroides') 
plt.title('Clusters de Películas')
plt.xlabel('X1: Presupuesto ($)')
plt.ylabel('X2: Ingresos ($)')
plt.legend()
plt.show()

