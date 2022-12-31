import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Waiter Tips Prediction\tips.csv")
print(data.head())

# tips given to the waiters according to:
# the total bill paid
# number of people at a table
# and the day of the week
figure = px.scatter(data_frame = data, x="total_bill", y="tip", size="size", color= "day", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill", y="tip", size="size", color= "sex", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="total_bill", y="tip", size="size", color= "time", trendline="ols")
figure.show()

# tips given to the waiters according to the days to find out which day the most tips are given to the waiters
figure = px.pie(data, values='tip', names='day',hole = 0.5)
figure.show()

# number of tips given to waiters by gender of the person paying the bill to see who tips waiters the most
figure = px.pie(data, values='tip', names='sex',hole = 0.5)
figure.show()

# smoker tips more or a non-smoker
figure = px.pie(data, values='tip', names='smoker',hole = 0.5)
figure.show()

# tips are given during lunch or dinner
figure = px.pie(data, values='tip', names='time',hole = 0.5)
figure.show()

# some data transformation by transforming the categorical values into numerical values
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()

# split the data into training and test sets
x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# train a machine learning model for the task of waiter tips prediction
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)
