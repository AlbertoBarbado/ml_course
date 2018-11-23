#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Decision Tree Classifier

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling -> No hace falta, ya que no se basa en distancias Euclideas
# pero se puede utilizar para agilizar los cálculos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Entrenamiento del modelo
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', # criterio usado
                                    max_leaf_nodes = None, # si se quiere definir un maximo de hojas por nodo
                                    min_samples_split = 2, # numero minimo de datos para hacer un split
                                    max_features = None, # si se quiere definir un limite de features usadas
                                    random_state = 0)
classifier.fit(X_train, y_train)

# Prediccion
y_pred = classifier.predict(X_test)

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importance
importance = classifier.feature_importances_


### Visualización
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
    plt.title('Decision Tree Classifier')
    plt.show()
    
# Train
visualize_classifier(classifier, X_train, y_train)

# Test
visualize_classifier(classifier, X_test, y_test)