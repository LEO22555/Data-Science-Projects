import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Salary Prediction\Salary_Data.csv")
print(data.head())

print(data.isnull().sum())

# relationship between the salary and job experience of the people
figure = px.scatter(data_frame = data, x="Salary",y="YearsExperience", size="YearsExperience", trendline="ols")
figure.show()

# splitting the data into training and test sets before training the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Training the ML model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predicting the salary of a person using the trained Linear regression model
a = float(input("Years of Experience : "))
features = np.array([[a]])
print("Predicted Salary = ", model.predict(features))

# Just found a perfect linear relationship between the salary and the job experience of the people. It means more job experience results in a higher salary.