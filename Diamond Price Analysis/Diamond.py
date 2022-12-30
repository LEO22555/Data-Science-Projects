import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Diamond Price Analysis\diamonds.csv")
print(data.head())

data = data.drop("Unnamed: 0",axis=1)

figure = px.scatter(data_frame = data, x="carat", y="price", size="depth", color= "cut", trendline="ols")
figure.show()

data["size"] = data["x"] * data["y"] * data["z"]
print(data)

figure = px.scatter(data_frame = data, x="size",y="price", size="size", color= "cut", trendline="ols")
figure.show()

fig = px.box(data, x="cut", y="price", color="color")
fig.show()

fig = px.box(data, x="cut", y="price", color="clarity")
fig.show()

correlation = data.corr()
print(correlation["price"].sort_values(ascending=False))

data["cut"] = data["cut"].map({"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5})

#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["carat", "cut", "size"]])
y = np.array(data[["price"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(xtrain, ytrain)

print("Diamond Price Prediction")
a = float(input("Carat Size: "))
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): "))
c = float(input("Size: "))
features = np.array([[a, b, c]])
print("Predicted Diamond's Price = ", model.predict(features))

# According to the diamond price analysis, we can say that the price and size of premium diamonds are higher than other types of diamonds.