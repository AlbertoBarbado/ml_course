# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:26:37 2018

@author: alber
"""

# K-Nearest Neighbors (K-NN)

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'euclidean', p = 2) # Defino una distancia que interesa, como la Euclidea
classifier.fit(X_train, y_train)

# Prediccion
y_pred = classifier.predict(X_test)

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

### Visualization
# Frontera de decision
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Datos como series
y_test_s = pd.Series(y_test)
target_names=['0','1']

# Indices de los labels
idxPlus=y_test_s[y_test_s==0].index
idxMin=y_test_s[y_test_s==1].index

# Visualizacion de los puntos de datos
plt.scatter(X_test[idxPlus,0],X_test[idxPlus,1],c='r',s=50, label='0')
plt.scatter(X_test[idxMin,0],X_test[idxMin,1],c='b',s=50, label='1')

plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()