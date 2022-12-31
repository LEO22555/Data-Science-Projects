import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Online Payments Fraud Detection\PS_20174392719_1491204439457_log.csv")
print(data.head())

print(data.isnull().sum())

# Exploring transaction type
print(data.type.value_counts())

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, values=quantity, names=transactions,hole = 0.5, title="Distribution of Transaction Type")
figure.show()

# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))