'''
En este script vamos a hacer un clúster para que la regresión se ajuste
mejor a los datos, ya que se han detectado 2 corportamientos diferenciados que
afectan al resultado de la emisión de CO2
'''

# Librerias
import pandas as pd
import numpy as np

# Análisis cluster
from sklearn.cluster import DBSCAN

def dbscan(X, epsilon=0.5, sample=4):
    # Esta función calculamos un DBSCAN
    db = DBSCAN(
        eps=epsilon,
        min_samples=sample
    ).fit(X)

    # Sacamos los labels
    labels = db.labels_

    return labels

def adjuntamos_labels(df, labels):
    df["cluster"] = labels
    return labels

def group_labels(df):

    # Agrupamos algunos cluster en 1
    df.loc[:,"cluster"].replace({2:1, 3:1, 4:1, 5:1}, inplace=True)
    # Eliminamos los cluster con valor -1
    df = df.loc[df["cluster"] != -1,:]
    return df


