
# import libraries

import numpy as np
import pandas as pd
from numpy import random
import math
random.seed(17)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

# TensorFlow y tf.keras
from tensorflow import keras

def model_f1_score_clasif(model, X_train, y_train, y_test):
    """retorna el F1_score del modelo de clasificación probado en el y_test"""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1_scored = metrics.f1_score(y_test, y_pred, average='macro')

    return f1_scored

def model_f1_score_reg(model, X_train, y_train, y_test):
    """retorna el F1_score del modelo de regresión probado en el y_test"""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # creamos un dataframe con las predicciones
    df_pred = pd.DataFrame(y_pred, columns=['prediccion'])
    # redondeamos la predicción
    df_pred['prediccion'] = round(df_pred['prediccion'], 0)
    # cambiamos el valor si sale del marco de las etiquetas (de 0 a 5)
    df_pred['prediccion'] = df_pred['prediccion'].apply(lambda x:
                                                        0 if x < 0
                                                        else
                                                        (5 if x > 5
                                                         else x))
    # cambiamos el tipo de la predicción a integer
    df_pred['prediccion'] = df_pred['prediccion'].astype(int)
    f1_scored = metrics.f1_score(y_test, df_pred['prediccion'], average='macro')

    return f1_scored

if __name__ == '__main__':

    # Import data
    df_train = pd.read_csv("datasets/train.csv")
    df_test = pd.read_csv("datasets/test.csv")
    df_train.drop('Unnamed: 0', axis=1, inplace=True)
    df_test.drop('Unnamed: 0', axis=1, inplace=True)

    # data analysis
    print(df_train.info())
    print(df_train.nunique())

    # no tenemos nulos
    # tenemos 3 variables string y binarias

    # cambiamos el nombre de las columnas por comodidad
    columnas = df_train.columns.str.replace(" ","_")
    df_train.columns = columnas
    columnas = df_test.columns.str.replace(" ","_")
    df_test.columns = columnas

    # intercambiamos el nivel de educación 0 y 1 para mantener una ordenalidad en el nivel de educación:
    # más alto el número, más alto el nivel
    df_train['parental_level_of_education'] = df_train['parental_level_of_education'].apply(lambda x:
                                                                                            0 if x == 1
                                                                                            else
                                                                                              (1 if x == 0
                                                                                               else x))

    # ploteamos la distribución de la target parental_level_of_education
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.countplot(data=df_train, x='parental_level_of_education')
    plt.show()

    # ploteamos la target según cada feature
    sns.scatterplot(data=df_train, y='parental_level_of_education', x='writing_score')
    plt.show()
    sns.scatterplot(data=df_train, y='parental_level_of_education', x='reading_score')
    plt.show()
    sns.scatterplot(data=df_train, y='parental_level_of_education', x='math_score')
    plt.show()

    # repartimos la target alrededor de su valor con una distribución normal para poder viusalizar mejor
    df_train['target_norm'] = np.random.normal(df_train['parental_level_of_education'], scale=0.3)

    # ploteamos la target_norm según cada feature
    sns.scatterplot(data=df_train, y='target_norm', x='writing_score')
    plt.savefig('relación nivel de educación parental VS nota de escritura del niño.png')
    plt.show()
    sns.scatterplot(data=df_train, y='target_norm', x='reading_score')
    plt.show()
    sns.scatterplot(data=df_train, y='target_norm', x='math_score')
    plt.show()

    # quitamos la variable 'target_norm'
    df_train.drop('target_norm', axis=1, inplace=True)

    # one hot encoding de las variables categóricas
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)

    # eliminamos las columnas redundantes
    df_train.drop(['gender_male', 'lunch_standard', 'test_preparation_course_none'], axis=1, inplace=True)
    df_test.drop(['gender_male', 'lunch_standard', 'test_preparation_course_none'], axis=1, inplace=True)

    # ploteamos las correlaciones
    plt.rcParams['figure.figsize'] = 8, 6
    sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
    plt.savefig('correlaciones.png')
    plt.show()

    # feature importance
    params = {'random_state': 42, 'n_jobs': 4, 'n_estimators': 100, 'max_depth': 4}
    # se entrena un RandomForest y se plotean las variables que tuvieron más importancia en el modelo
    y = df_train['parental_level_of_education']
    x = df_train.drop('parental_level_of_education', axis=1)
    # entrena un RandomForest Classifier
    clf = RandomForestClassifier(**params)
    clf = clf.fit(x, y)
    # Plotea las mejores Features importances
    features = clf.feature_importances_[:10]
    columnas = x.columns[:10]
    imp = pd.Series(data=features, index=columnas).sort_values(ascending=False)
    plt.figure(figsize=(10,8))
    plt.title("Feature importance")
    sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')
    plt.show()

    # split train/test
    # se guarda un 30% de datos para el test y un 70% para el train
    y = df_train['parental_level_of_education']
    X = df_train.drop('parental_level_of_education', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # se estandarizan las variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # estandariza y define la regla de estandarización
    X_test = scaler.transform(X_test)         # estandariza según la regla definida con el X_train
    df_test = scaler.transform(df_test)       # estandariza según la regla definida con el X_train

    # además se normalizan las variables
    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)   # normaliza y define la regla de normalización
    X_test = scaler.transform(X_test)         # normaliza según la regla definida con el X_train
    df_test = scaler.transform(df_test)         # normaliza según la regla definida con el X_train

    # creamos listas para almacenar el nombre de los modelos y los f1_scores
    modelos = []
    f1s = []

    # probamos diferentes modelos de clasificación multiclase
    modelo = 'LogisticRegression'
    model = LogisticRegression(random_state=42)
    f1_score = model_f1_score_clasif(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    modelo = 'RandomForestClassifier'
    model = RandomForestClassifier(random_state=42)
    f1_score = model_f1_score_clasif(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    modelo = 'GradientBoostingClassifier'
    model = GradientBoostingClassifier(random_state=42)
    f1_score = model_f1_score_clasif(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    modelo = 'RidgeClassifier'
    model = RidgeClassifier(random_state=42)
    f1_score = model_f1_score_clasif(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    # modelo = 'xgbClassifier'
    # model = xgb.XGBRFClassifier(random_state=42)
    # f1_score = model_f1_score_clasif(model, X_train, y_train, y_test)
    # modelos.append(modelo)
    # f1s.append(f1_score)

    # redes neuronales
    modelo = 'redes neuronales'
    num_etiquetas = len(y_train.unique())
    model = keras.Sequential([
        keras.layers.Dense(X_train.shape[0], activation='relu'),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dense(180, activation='relu'),
        keras.layers.Dense(num_etiquetas, activation='softmax')
    ])
    model.compile(optimizer='adamax',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, verbose=0)
    predictions = model.predict(X_test)
    # creamos un dataframe con las probabilidades de cada etiqueta
    df_comparacion = pd.DataFrame(predictions)
    # creamos la columna con la etiqueta más probable
    df_comparacion["top_probable"] = df_comparacion.apply(lambda x:
                                                          np.argmax([x[0], x[1], x[2], x[3], x[4], x[5]]), axis=1)
    f1_score = metrics.f1_score(y_test, df_comparacion['top_probable'], average='macro')
    modelos.append(modelo)
    f1s.append(f1_score)

    # probamos con algunos modelos de regresión

    modelo = 'LinearRegression'
    model = LinearRegression()
    f1_score = model_f1_score_reg(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    modelo = 'Ridge regression'
    model = Ridge()
    f1_score = model_f1_score_reg(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    modelo = 'RandomForestRegressor'
    model = RandomForestRegressor()
    f1_score = model_f1_score_reg(model, X_train, y_train, y_test)
    modelos.append(modelo)
    f1s.append(f1_score)

    # imprimimos el resultado de los modelos
    dict_f1s = {'modelo': modelos,
                'f1_score': f1s}
    df_f1s = pd.DataFrame(dict_f1s, columns=['modelo', 'f1_score'])
    print(df_f1s.sort_values('f1_score', ascending=False))

    # hiperparametrizamos el mejor modelo de clasificación a ver si supera las redes neuronales
    model = GradientBoostingClassifier(n_estimators=900, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')
    print(f1_score)

    # hacemos la predicción del test
    predicciones = model.predict(df_test)
    df_pred = pd.DataFrame(predicciones, columns=['predicciones'])

    # volvemos a intercambiar el nivel de educación 0 y 1 para volver a las etiquetas originales
    df_pred['predicciones'] = df_pred['predicciones'].apply(lambda x:
                                                                                            0 if x == 1
                                                                                            else
                                                                                            (1 if x == 0
                                                                                             else x))

    # se exporta a csv
    df_pred.to_csv('predictions.csv', index=False)
    # se exporta a json
    df_pred.to_json('predictions.json')


