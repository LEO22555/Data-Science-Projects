import pandas as pd
import numpy as np
data = pd.read_csv(r"https://raw.githubusercontent.com/amankharwal/Website-data/master/supplement.csv")
data.head()

data.info()
data.isnull().sum()
data.describe()

# knowing about the factors affecting the number of orders for supplements
import plotly.express as px
pie = data["Store_Type"].value_counts()
store = pie.index
orders = pie.values

fig = px.pie(data, values=orders, names=store)
fig.show()

# distribution of the number of orders, according to the location
pie2 = data["Location_Type"].value_counts()
location = pie2.index
orders = pie2.values

fig = px.pie(data, values=orders, names=location)
fig.show()

# looking at the distribution of the number of orders, according to the discount
pie3 = data["Discount"].value_counts()
discount = pie3.index
orders = pie3.values

fig = px.pie(data, values=orders, names=discount)
fig.show()

# how holidays affect the number of orders
pie4 = data["Holiday"].value_counts()
holiday = pie4.index
orders = pie4.values

fig = px.pie(data, values=orders, names=holiday)
fig.show()

# changing some of the string values to numerical values
data["Discount"] = data["Discount"].map({"No": 0, "Yes": 1})
data["Store_Type"] = data["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
data["Location_Type"] = data["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})
data.dropna()

x = np.array(data[["Store_Type", "Location_Type", "Holiday", "Discount"]])
y = np.array(data["#Order"])

# splitting the data into 80% training set and 20% test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

#  using light gradient boosting regression algorithm to train the model (lightgbm)
import lightgbm as ltb
model = ltb.LGBMRegressor()
model.fit(xtrain, ytrain)

# having a look at the predicted values
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Orders": ypred.flatten()})
print(data.head())

