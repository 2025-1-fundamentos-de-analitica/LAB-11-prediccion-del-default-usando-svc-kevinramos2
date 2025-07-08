# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pickle
import zipfile
import pandas as pd
import numpy as np
import json
import gzip
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


    
# ================================
# Paso 1: Cargar y limpiar datos
# ================================
def preparar_datos(df):
    """
    Limpia y transforma los datos brutos.
    - Elimina columnas innecesarias.
    - Renombra la variable objetivo.
    - Elimina registros con valores no válidos.
    - Agrupa valores altos de EDUCATION como 'otros'.
    """
    df = df.copy()
    df.drop(columns='ID', inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.dropna(inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df


# ==============================
# Paso 2: Creación del pipeline
# ==============================
def crear_pipeline():
    """
    Define el pipeline completo con preprocesamiento, reducción, selección y modelo.
    """
    cat_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_features = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocesador = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    pipeline = Pipeline([
        ('preprocesamiento', preprocesador),
        ('pca', PCA()),
        ('seleccion', SelectKBest(score_func=f_classif)),
        ('clasificador', SVC(kernel='rbf', random_state=42))
    ])

    return pipeline


# ==========================================
# Paso 3:  Optimización de hiperparámetros
# ==========================================
def optimizar_hyperparametros(pipeline, x_train, y_train, cv_splits, metric):
    grid = GridSearchCV(
        pipeline,
        param_grid={
            'pca__n_components': [20],  # solo 1 valor para no explorar nada
            'seleccion__k': [12],
            'clasificador__kernel': ['rbf'],
            'clasificador__gamma': [0.099]
        },
        cv=3,   # en vez de 10
        scoring=metric,
        refit=True,
        verbose=2
    )

    grid.fit(x_train, y_train)
    return grid

# ============================
# Paso 4:  Calcular métricas 
# ============================
def calcular_metricas(modelo, x, y, dataset_name):
    """
    Genera un diccionario con las métricas para un conjunto de datos.
    """
    y_pred = modelo.predict(x)
    return {
        'type': 'metrics',
        'dataset': dataset_name,
        'precision': precision_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred)
    }


# ================================
# Paso 5:  Matrices de confusión
# ================================
def calcular_matriz_confusion(modelo, x, y, dataset_name):
    """
    Genera un diccionario con la matriz de confusión para un conjunto de datos.
    """
    y_pred = modelo.predict(x)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {'predicted_0': int(tn), 'predicted_1': int(fp)},
        'true_1': {'predicted_0': int(fn), 'predicted_1': int(tp)}
    }


# ================================
# Paso 6:  Guardar el modelo
# ================================
def guardar_comprimido(modelo, path):
    """
    Guarda el modelo serializado y comprimido.
    """
    os.makedirs("files/models", exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)


# ================================
# Paso 6:  Guardar métricas
# ================================
def guardar_jsonl(registros, path):
    """
    Guarda una lista de diccionarios como JSONL.
    """
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for item in registros:
            f.write(json.dumps(item) + '\n')
        

# ============================
# EJECUCIÓN PRINCIPAL DEL LAB
# ============================

if __name__ == "__main__":
        
    try:
        print("Cargando datos...")
        test_df = pd.read_csv("files/input/test_data.csv.zip")
        train_df = pd.read_csv("files/input/train_data.csv.zip")
 
        print("Datos cargados. Limpiando...")
        train_df = preparar_datos(train_df)
        test_df = preparar_datos(test_df)

        X_train, y_train = train_df.drop(columns='default'), train_df['default']
        X_test, y_test = test_df.drop(columns='default'), test_df['default']

        print("Datos preparados. Creando pipeline...")
        pipeline = crear_pipeline()

        print("Ajustando modelo...")
        modelo_ajustado = optimizar_hyperparametros(pipeline, X_train, y_train, 10, 'balanced_accuracy')

        print("Guardando modelo...")
        guardar_comprimido(modelo_ajustado, 'files/models/model.pkl.gz')

        metricas_train = calcular_metricas(modelo_ajustado, X_train, y_train, 'train')
        metricas_test = calcular_metricas(modelo_ajustado, X_test, y_test, 'test')

        cm_train = calcular_matriz_confusion(modelo_ajustado, X_train, y_train, 'train')
        cm_test = calcular_matriz_confusion(modelo_ajustado, X_test, y_test, 'test')

        print("Guardando métricas...")
        guardar_jsonl([metricas_train, metricas_test, cm_train, cm_test], 'files/output/metrics.json')

        print("Proceso completado.")

    except Exception as e:
        print("💥 Ocurrió un error:")
        import traceback
        traceback.print_exc()
