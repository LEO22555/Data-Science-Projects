
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Currency Exchange Rate Prediction\COP=X.csv")
print(data.head())

plt.figure(figsize=(10, 4))
plt.title("COP - USD Exchange Rate")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()

# correlation between the features before training the currency exchange rate prediction model
print(data.corr())
sns.heatmap(data.corr())
plt.show()

# storing the most relevant features in the variable x and storing the target column in the variable y
x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

# splitting the dataset and train a currency exchange prediction model using the Decision Tree Regression model
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# predicted values of currency exchange rates of Colombian pesos for the next 5 days
data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(data.head())