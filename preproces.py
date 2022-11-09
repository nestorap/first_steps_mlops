'''
En este notebook vamos a agrupar los procesos de leer y transformar el dataset
'''

# Importamos librerías

# Tratamiento de datos y matrices
import pandas as pd
import numpy as np

# Transformación de los datos
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_data():
    df = pd.read_csv('data/coches.csv')
    return df

def seleccion_columnas(df):
    df1 = df.copy()
    
    X = df1.loc[:,['FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
    y = df1.loc[:,"CO2EMISSIONS"]

    # Escalamos los datos de X
    X = MinMaxScaler().fit_transform(X)

    return X, y



