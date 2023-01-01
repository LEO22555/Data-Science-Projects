import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())

print(data.isnull().sum())

import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales", y="TV", size="TV", trendline="ols")
figure.show()

# visualizing the relationship between the amount spent on advertising on newspapers and units sold
figure = px.scatter(data_frame = data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols", title="Relationship between the amount spent on advertising on newspapers and units sold")
figure.show()

# relationship between the amount spent on advertising on radio and units sold
figure = px.scatter(data_frame = data, x="Sales", y="Radio", size="Radio", trendline="ols", title="Relationship between the amount spent on advertising on radio and units sold")
figure.show()

# correlation of all the columns with the sales column
correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))

# spliting the data into training and test sets
x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# training the model to predict future sales
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# predicting how many units of the product can be sold based on the amount spent on its advertising on various platforms
#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))
