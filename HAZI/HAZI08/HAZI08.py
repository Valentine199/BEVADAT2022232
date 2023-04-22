import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


'''
Készíts egy függvényt, betölti majd vissza adja az iris adathalmazt.
Egy példa a kimenetre: iris
return type: sklearn.utils.Bunch
függvény neve: load_iris_data
1
'''


def load_iris_data() -> sklearn.utils.Bunch:
    return load_iris()


iris = load_iris_data()
#print(iris)
'''
Készíts egy függvényt, ami a betölti az virágokhoz tartozó levél méretket egy dataframebe, majd az első 5 sort visszaadja.
Minden oszlop tartalmazza, hogy az milyen mérethez tartozik.
Egy példa a bemenetre: iris
Egy példa a kimenetre: iris_df
return type: pandas.core.frame.DataFrame
függvény neve: check_data
1
'''


def check_data(iris) -> pd.core.frame.DataFrame:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df.head()

#print(check_data(iris))
''' 
Készíts egy függvényt ami előkészíti az adatokat egy lineaáris regressziós model feltanításához.
Featurejeink legyenek a levél méretek kivéve a "sepal length (cm)", ez legyen a targetünk.
Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
1
'''

def linear_train_data(iris):
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = df[["sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    y = df["sepal length (cm)"]
    return X, y

linX, linY = linear_train_data(iris)

''' 
Készíts egy függvényt ami előkészíti az adatokat egy logisztikus regressziós model feltanításához.
Featurejeink legyenek a levél méretek, targetünk pedig a 0, 1-es virág osztályok.
Fontos csak azokkal az adatokkal tanítsunk amihez tartozik adott target. 
Egy példa a bemenetre: iris
Egy példa a kimenetre: X, y
return type: (numpy.ndarray, numpy.ndarray)
0
'''


def logistic_train_data(iris):
    df = pd.DataFrame(iris.data)
    df['target'] = iris.target
    df= df[df['target'].notna()]

    df.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", 'target']
    X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    mask = (df["target"] < 2) & (df["target"] >= 0)
    y_df = df[mask]
    y = y_df.iloc[:, -1]
    return X, y

#logX, logy = logistic_train_data(iris)

'''
Készíts egy függvényt ami feldarabolja az adatainkat train és test részre. Az adatok 20%-át használjuk fel a teszteléshez.
Tegyük determenisztikussá a darabolást, ennek értéke legyen 42.
Egy példa a bemenetre: X, y
Egy példa a kimenetre: X_train, X_test, y_train, y_test
return type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
1
'''


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = split_data(linX, linY)
#X_train, X_test, y_train, y_test = split_data(logX, logy)
'''
Készíts egy függvényt ami feltanít egy lineaáris regressziós modelt, majd visszatér vele.
Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LinearRegression
1
'''


def train_linear_regression(X_train, y_train) -> LinearRegression:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


'''
Készíts egy függvényt ami feltanít egy logisztikus regressziós modelt, majd visszatér vele.
Egy példa a bemenetre: X_train, y_train
Egy példa a kimenetre: model
return type: sklearn.linear_model._base.LogisticRegression
1
'''


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


''' 
Készíts egy függvényt, ami a feltanított modellel predikciót tud végre hajtani.
Egy példa a bemenetre: model, X_test
Egy példa a kimenetre: y_pred
return type: numpy.ndarray
1
'''


def predict(model, X_test) -> numpy.ndarray:
    if model is LogisticRegression:
        return model.predict(X_test)
    else:
        return model.predict(X_test)


'''
Készíts egy függvényt, ami vizualizálni tudja a label és a predikciók közötti eltérést.
Használj scatter plotot a diagram elkészítéséhez.
Diagram címe legyen: 'Actual vs Predicted Target Values'
Az x tengely címe legyen: 'Actual'
Az y tengely címe legyen: 'Predicted'
Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: scatter plot
return type: matplotlib.figure.Figure
1
'''


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_title('Actual vs Predicted Target Values')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    return fig


''' 
Készíts egy függvényt, ami a Négyzetes hiba (MSE) értékét számolja ki a predikciók és a valós értékek között.
Egy példa a bemenetre: y_test, y_pred
Egy példa a kimenetre: mse
return type: float
1
'''


def evaluate_model(y_test, y_pred):
    return mean_squared_error(y_test, y_pred)
