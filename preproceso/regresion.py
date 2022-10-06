'''
En este script vamos a agrupar las funciones que nos permitan entrenar el modelo de ml
Aquí vamos a tener 2 modelos, un random forest para ver en que cluster se corresponderia
el nuevo caso y una regresion lineal para ver cuanto de CO2 emite
'''

# Librerías
import pandas as pd
import numpy as np

# ml
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def selecion_columna_clasif(df):

    df_clasif = df.drop(['FUELCONSUMPTION_HWY', 'CO2EMISSIONS', 'MODELYEAR', 'MODEL']])
    return df_clasif

def numerical(df_clasif):
    # Las columnas categóricas las volvemos numéricas para el clasificador
    for i in ('MAKE', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'):
        keys = df_clasif.loc[:,i].unique() # Guardamos los valores únicos de la columna
        valores = range(len(keys)) # Medimos el número de valores para asignarle a cada valor un número
        dic = dict(zip(keys, valores))
        df_clasif.loc[:,i].replace(dic, inplace=True)

    # Separamos las columnas X e y
    X = df_clasif.values
    y = df.loc[:,"cluster"]

    return X, y

def preproces_clasif(X):
    #preprocesamos la columna X

    madmax = MinMaxScaler()
    madmax.fit(X)
    X = madmax.transform(X)

    return X

def train_clasif(X, y):

    skf = StratifieldkFold(n_splits=5)

    for train_index, text_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)





