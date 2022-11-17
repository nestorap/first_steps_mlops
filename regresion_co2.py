'''
Introducción -> En este script se almacena el código que nos saca un modelo de regresión.
La idea es desplegar este modelo en github y hacer uso de github actions para que se reentrene y nos pinte unos gráficos
que nos indique la calidad del modelo.

En principio, será algo estático, y esto tiene un enfoque de ejercicio y de dar primeros pasos
'''

#### Importamos las librerías #####

# Tratamiento de datos
import pandas as pd
import numpy as np

# Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías para ml
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

import pickle

from preproces import read_data

import logging
import sys

logging.basicConfig(
    format = "%(asctime)s %(levelname)s:%(name)s : %(message)s",
    level = logging.INFO,
    datefmt = "%H:%M:%S",
    stream = sys.stderr
)

logger = logging.getLogger(__name__)


# Leemos los datos
logger.info("Cargamos el datast")
df = read_data()

logger.info("mostramos un head del dataset")
print(df.head())

# Separamos en X e y
logger.info("Separamos en X e y")
X = df.loc[:,'FUELCONSUMPTION_HWY'].values.reshape(-1, 1)
y = df.loc[:,"CO2EMISSIONS"].values

# Separamos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cargamos el modelo y entrenamos
logger.info("Entrenamos el modelo")
regresion = linear_model.LinearRegression()
regresion.fit(X_train, y_train)


# Sacamos el score de train
train_score = regresion.score(X_train, y_train) * 100
# Sacamos el score de test
test_score = regresion.score(X_test, y_test) * 100

# Guardamos los scores en un txt
with open("metrics.txt", "w") as outfile:
    outfile.write("Varianza explicada de train: %2.1f%%\n" % train_score)
    outfile.write("Varianza explicada de test %2.1f%%\n" % test_score)


######### Graficamos #############

# Sacamos los coeficientes del modelo para plotear la linea que nos arroja la regresion lineal
coef = regresion.coef_
intercept = regresion.intercept_

# Sacamos un plot de los datos y de la linea
logger.info("ploteamos los datos y el resultado")
sns.scatterplot(x=X.flatten(), y=y.flatten(), color="magenta")
# Ploteamos la linea
plt.plot(X_train, coef[0] * X_train + intercept, "b")
plt.tight_layout()
plt.savefig("resultado.png", dpi=120)
plt.close()

# Sacamos plot de los residuos
y_pred = regresion.predict(X_test) +np.random.normal(0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns=["True", "Pred"])

axis_fs = 18 # Fontsize
title_fs = 22 # Fontsize

ax = sns.scatterplot(x="True", y="Pred", data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True CO2 emission', fontsize=axis_fs)
ax.set_ylabel('Prediccion CO2 emission', fontsize=axis_fs)
ax.set_title('Residuals', fontsize=title_fs)

ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5, 8.5))
plt.xlim((2.5, 8.5))

plt.tight_layout()
plt.savefig("residuos.png", dpi=120)

# Guardamos el modelo
#logger.info("Guardamos el modelo")
#pickle.dump(regresion, open("first_model", "wb"))





