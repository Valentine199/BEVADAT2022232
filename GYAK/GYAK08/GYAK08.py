import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from LinearRegressionSkeleton import LinearRegression

iris = load_iris()
lin_reg = LinearRegression(lr=0.1)

df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = df["petal width (cm)"].values
y = df['sepal length (cm)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg.fit(X_train, y_train)

preds = []
for x in X_test:
    preds.append(lin_reg.predict(x))


print("Mean Absolute Error:", np.mean(np.abs(preds - y_test)))
print("Mean Squared Error:", np.mean((preds - y_test)**2))

plt.scatter(X_test, y_test)
plt.plot([min(X_test), max(X_test)], [min(preds), max(preds)], color='red') # predicted
plt.show()