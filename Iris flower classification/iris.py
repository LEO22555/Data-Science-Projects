import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Iris flower classification\IRIS.csv")

print(iris.head())
print(iris.describe())

# Quick look at the target labels
print("Target Labels", iris["species"].unique())

# plotting the data using a scatter plot which will plot the iris species according to the sepal length and sepal width
import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

# split the data into training and test sets, and then I will use the KNN classification algorithm to train the iris classification model
x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

# input a set of measurements of the iris flower and use the model to predict the iris species
x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
