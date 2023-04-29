# imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits


class KMeansOnDigits():

    def __init__(self, n_clusters=10, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state


    # Készíts egy függvényt ami betölti a digits datasetet
    # NOTE: használd az sklearn load_digits-et
    # Függvény neve: load_digits()
    # Függvény visszatérési értéke: a load_digits visszatérési értéke
    # 1
    def load_dataset(self):
        self.digits = load_digits()

    # Készíts egy függvényt ami létrehoz egy KMeans model-t 10 db cluster-el
    # NOTE: használd az sklearn Kmeans model-jét (random_state legyen 0)
    # Miután megvan a model predict-elj vele
    # NOTE: használd a fit_predict-et
    # Függvény neve: predict(n_clusters:int,random_state:int,digits)
    # Függvény visszatérési értéke: (model:sklearn.cluster.KMeans,clusters:np.ndarray)
    # 4
    def predict(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        X = self.digits.data
        y = self.digits.target
        self.clusters = self.model.fit_predict(X)

    # Készíts egy függvényt ami visszaadja a predictált cluster osztályokat
    # NOTE: amit a predict-ből visszakaptunk "clusters" azok lesznek a predictált cluster osztályok
    # HELP: amit a model predictált cluster osztályok még nem a labelek, hanem csak random cluster osztályok,
    #       Hogy label legyen belőlük:
    #       1. készíts egy result array-t ami ugyan annyi elemű mint a predictált cluster array
    #       2. menj végig mindegyik cluster osztályon (0,1....9)
    #       3. készíts egy maszkot ami az adott cluster osztályba tartozó elemeket adja vissza
    #       4. a digits.target-jét indexeld meg ezzel a maszkkal

    #       5. számold ki ennel a subarray-nek a móduszát
    #       6. a result array-ben tedd egyenlővé a módusszal azokat az indexeket ahol a maszk True
    #       Erre azért van szükség mert semmi nem biztosítja nekünk azt, hogy a "0" cluster a "0" label lesz, lehet, hogy az "5" label lenne az.

    # Függvény neve: get_labels(clusters:np.ndarray, digits)
    # Függvény visszatérési értéke: labels:np.ndarray
    # 7

    def get_labels(self):
        result = np.empty(self.clusters.shape)
        for c in self.digits.target_names:
            mask = self.clusters == c
            sub_arr = self.digits.target[mask]
            arr_mod = mode(sub_arr).mode.item()

            result[mask] = arr_mod

        self.labels = result


    # Készíts egy függvényt ami kiszámolja a model accuracy-jét
    # Függvény neve: calc_accuracy(target_labels:np.ndarray,predicted_labels:np.ndarray)
    # Függvény visszatérési értéke: accuracy:float
    # NOTE: Kerekítsd 2 tizedes jegyre az accuracy-t

    def calc_accuracy(self):
        acc = accuracy_score(self.digits.target, self.labels)
        self.accuracy = np.round(acc, 2)

    # Készíts egy confusion mátrixot és plot-old seaborn segítségével
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)

#meanes = KMeansOnDigits()
#meanes.load_dataset()

#dig = meanes.digits

# Vizsgáld meg a betöltött adatszetet (milyen elemek vannak benne stb.)
# 2
#print(dig.feature_names)
#print(dig.target_names)

# Vizsgáld meg a data paraméterét a digits dataset-nek (tartalom,shape...)
# 3
#dat = dig.data
#print(dat)
#print(dat.shape)


#meanes.predict()
#clusteres = meanes.model
# Vizsgáld meg a shape-jét a kapott model cluster_centers_ paraméterének.
# 5
#print(clusteres.cluster_centers_.shape)

# Készíts egy plotot ami a cluster középpontokat megjeleníti
# 6
#plt.scatter(clusteres.cluster_centers_[0], clusteres.cluster_centers_[1])
#plt.show()

#meanes.get_labels()
#meanes.calc_accuracy()
#meanes.confusion_matrix()
