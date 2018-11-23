# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:51:58 2018

@author: alber
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn import datasets # datasets disponibles en sklearn
data = datasets.load_boston() # carga del dataset BOSTON

# Informacion del dataset
print(data.DESCR)

# Variables independientes (entrada)
df = pd.DataFrame(data.data, columns=data.feature_names)

# y = Valor medio de casas ocupadas por sus propietarios en 1000 dolares.
# X1 = CRIM: representa el crimen per c ́apita por ciudad.
# X2 = ZN: la proporci ́on de zonas residenciales en un  ́area determinada.
# X3 = INDUS: la proporci ́on de acres dedicada a negocios al por menor en la ciudad.
# X4 = CHAS: variable binaria (=1 si las v ́ıas cruzan el r ́ıo y 0 en otro caso).
# X5 = NOX: concentraci ́on de  ́oxido n ́ıtrico (partes por mill ́on).
# X6 = RM: n ́umero medio de habitaciones por vivienda.
# X7 = AGE: proporci ́on de edificios ocupados por sus propietarios, construidos antes de 1940.
# X8 = DIS: representa la distancia ponderada a cinco centros de empleo en Boston.
# X9 = RAD:  ́ındice de accesibilidad a las autopistas radiales.
# X10 = TAX: valor total de la tasa de impuestos por 10.000 d ́olares.
# X11 = PTRATIO: representa el ratio alumno-profesor por ciudad.
# X12 = B: valor definido como 1000(Bk - 0.63)^2 donde Bk es la proporcion de afroamericanos en la ciudad
# X13 = LSTAT: porcentaje de clase baja en la poblacion

# Variable de salida
target = pd.DataFrame(data.target, columns=["MEDV"])
y = target["MEDV"]

### Modelo con todas las variables
#X = df[["RM", "LSTAT"]]
X = df # todas las variables

# Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()

# Se descartan INDUS, NOX, AGE
X = df.copy()
X.drop(['INDUS', 'NOX', 'AGE'], axis=1, inplace=True)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()

# predicciones de la salida
y_pred = model.predict(X_test)

# Metricas de evaluacion
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)

# Correlación
import seaborn as sns
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)



### Modelo con scikit-learn

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))

# Modelo
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediccion de valores
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
y_test = y_test.values.reshape(-1, 1)

# Metricas de evaluacion
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)
