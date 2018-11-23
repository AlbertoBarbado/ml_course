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


### Silhouette Score
from sklearn.cluster import KMeans
from sklearn import metrics

def silhouette_selection(referencia, figure=False):
    """
    En este caso se usa un valor de referencia para detenerse cuando la diferencia de usar un k u otro
    sea muy significativa (ya que querria decir que ha empeorado mucho el modelo)
    
    """
    shc = [] # Pongo un vector a 0 para ver los distintos scores segun el numero de clusters que defina
    diff = 0
    i = 1
    valores = []
    
    while abs(diff) < referencia:
        i += 1 # Esta metrica necesita al menos 2 clusters
        valores.append(i)
        print("Iteracion Nº Clusters: k: {k}".format(k=i))
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        score = metrics.silhouette_score(X, kmeans.labels_, metric="euclidean", sample_size=len(X))
        # Primera iteracion
        if i == 2:
            pass
        # Resto de iteraciones
        else:
            diff = (shc[-1] - score)/shc[-1]
        shc.append(score)
        print("Silhouette score = {0} para Nº Clusters {1}".format(score, i))
        print("Diferencia con el score anterior", diff)
    
    if figure:
        plt.figure()
        plt.bar(valores, shc, width=0.7, color='blue', align='center')
        plt.title('Silhouette Score vs Numero de Clusters')
        plt.show()
    
    # Clusters finales
    k = i-1
    return shc, k
    

# Obtencion de k optima
referencia = 0.15
shc, k = silhouette_selection(referencia, figure=True)

# K-means
kmeans = KMeans(n_clusters = k, init= 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X) 

### Visualizar clusters (se pintan uno a uno)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'C1') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'C2')

# Para pintar los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroides') 
plt.title('Clusters de Clientes')
plt.xlabel('X1: Balance en la Cuenta ($)')
plt.ylabel('X2: Gastos en Compras ($)')
plt.legend()
plt.show()