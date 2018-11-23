# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:49:40 2018

@author: alber
"""

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###############################################################################

### Data Preprocessing

###############################################################################

# Cargar dataset
df = pd.read_csv('Salary_Data.csv')

# Visualizar los datos
df.plot(x='YearsExperience', y='Salary', title="Evolucion del Salario segun los Años de Experiencia") # Se ve relacion lineal

# Separacion en variables entrada/salida
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



###############################################################################

### Comprobacion del modelo

###############################################################################


#### P-Valores
# Ver p-valores
import statsmodels.api as sm
#X_train = sm.add_constant(X_train) # b0
#X_test = sm.add_constant(X_test) # b0
model = sm.OLS(y_train, X_train).fit()
model.summary()


#### Linealidad
# Comprobacion de linealidad
import statsmodels.stats.api as sms
sms.linear_harvey_collier(model)
# p-valor = 0.272 > 0.05 -> No se rechaza H0=hay linearidad

#### Normalidad Residuos
# Obtencion residuos
from statsmodels.compat import lzip
residuos = model.resid

# Histogramas
plt.hist(residuos, range=(-45000, 45000))

# Q-Q Plot
import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residuos, plot=ax, fit=True)
print(r**2)
# r^2 = 0.957
# Se ve normalidad

# Test D'Agostino
from scipy.stats import normaltest
normaltest(residuos)
# p-valor = 0.808 >> 0.05 -> No se rechaza H0=datos normales

# Test Jarque-Bera
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(residuos)
lzip(name, test)

# Omni Test
name = ['Chi^2', 'Two-tail probability']
test = sms.omni_normtest(residuos)
lzip(name, test)


### Comprobacion Homocedasticidad
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(X_train, residuos)

#
## Breush-Pagan test:
#name = ['Lagrange multiplier statistic', 'p-value', 
#        'f-value', 'f p-value']
#test = sms.het_breushpagan(model.resid, model.model.exog)
#lzip(name, test)

# Goldfeld-Quandt test
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)
# model.model.exog -> Parametros de entrada
# model.model.endog -> Parametros de salida
# p-valor=0.78 >> 0.05 -> No se puede rechazar H0=hay homocedasticidad

### Autocorrelacion errores
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(residuos))
# Valor cercano a 2 -> No correlacion
# El valor esta acotado entre 0 (max. correlacion positiva) y 4 (max. correlacion negativa)


### Comprobacion de influencia de posibles outliers
from statsmodels.stats.outliers_influence import OLSInfluence
test_class = OLSInfluence(model)
test_class.dfbetas[:5,:]
# DFBetas da la diferencia en cada parametro estimado con y sin los puntos de influencia. Hay un DFBEta por cada
# datapoint; asi, con n observaciones y k variables hay n*k DFBEtas
# Se puede utilizar 2 o 2/sqrt(n) como umbral para ver que DFBetas son significativo y por lo tanto tienen mucha influencia 
 
from statsmodels.graphics.regressionplots import plot_leverage_resid2, influence_plot
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(model, ax = ax)
influence_plot(model)
# Hay que tener cuidado con las observaciones que tengan un leverage alto y unos residuos altos ya que son las que mas
# van a estar influyendo en el modelo
# En el diagrama de influencia no se aprecia nada en el cuadrante superior derecho, pero en el de los residuos
# al cuadrado se observa como la observacion 7 (x=10.5) influye sensiblemente en la generalidad del modelo
# como es un problema ejemplo se va a dejar aunque se deberia tener precaucion con ella


# Parametros del modelo
p =  model.params
print(p)

# Metricas
model.aic

###############################################################################

### Predicciones del modelo

###############################################################################

# predicciones de la salida
y_pred = model.predict(X_test)

# Metricas de evaluacion
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)
# R2 remarca la explicabilidad del modelo. Cuanto mas cercano sea a 1 el modelo consigue
# explicar/expresar mejor los datos. Valor entre 0 y 1.
# 0.8111 es buen resultado -> 81.1% explicado

# Visualizing the Training results
plt.scatter(X_train, y_train, color = 'red') #Pinto en rojo los valores reales de train como puntos
plt.plot(X_train, model.predict(X_train), color = 'blue') #Pinto en azul y en forma de linea la predicción sobre mis valores de train ya que aplico la regresión sobre esos mismos valores de train, mi modelo entrenado sobre esos valores
plt.title('Salary vs Esperience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red') #Pinto ahora los puntos de test sobre el mismo modelo que he construido para validarlo
plt.plot(X_train, model.predict(X_train), color = 'blue') #Pinto en azul y en forma de linea la predicción sobre mis valores de train ya que aplico la regresión sobre esos mismos valores de train, mi modelo entrenado sobre esos valores
plt.title('Salary vs Esperience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


###############################################################################

### Mejoras en el modelo

###############################################################################


### Añadiendo un b0
import statsmodels.api as sm
X_train = sm.add_constant(X_train) # b0
X_test = sm.add_constant(X_test) # b0
model = sm.OLS(y_train, X_train).fit()
model.summary()

# predicciones de la salida
y_pred = model.predict(X_test)


# Visualizing the Training results
plt.scatter(X_train[:,1], y_train, color = 'red') #Pinto en rojo los valores reales de train como puntos
plt.plot(X_train[:,1], model.predict(X_train), color = 'blue') #Pinto en azul y en forma de linea la predicción sobre mis valores de train ya que aplico la regresión sobre esos mismos valores de train, mi modelo entrenado sobre esos valores
plt.title('Salary vs Esperience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test[:,1], y_test, color = 'red') #Pinto ahora los puntos de test sobre el mismo modelo que he construido para validarlo
plt.plot(X_train[:,1], model.predict(X_train), color = 'blue') #Pinto en azul y en forma de linea la predicción sobre mis valores de train ya que aplico la regresión sobre esos mismos valores de train, mi modelo entrenado sobre esos valores
plt.title('Salary vs Esperience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Metricas
model.aic

### Escalando las variables de entrada

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
#
#import statsmodels.api as sm
#model = sm.OLS(y_train, X_train).fit()
#model.summary()
#
#
## Obtencion residuos
#from statsmodels.compat import lzip
#residuos = model.resid
#
## Histogramas
#plt.hist(residuos, range=(-5000, 5000))
#
## predicciones de la salida
#y_pred = model.predict(X_test)


