import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Student Marks Prediction\Student_Marks.csv")
print(data.head(10))

print(data.isnull().sum())
data["number_courses"].value_counts()

# Number of Courses and Marks Scored
figure = px.scatter(data_frame=data, x = "number_courses", y = "Marks", size = "time_study", title="Number of Courses and Marks Scored")
figure.show()

# Time Spent and Marks Scored
figure = px.scatter(data_frame=data, x = "time_study", y = "Marks", size = "number_courses", title="Time Spent and Marks Scored", trendline="ols")
figure.show()

# correlation between the marks scored by the students and the other two columns in the data
correlation = data.corr()
print(correlation["Marks"].sort_values(ascending=False))

# splitting the data into training and test sets
x = np.array(data[["time_study", "number_courses"]])
y = np.array(data["Marks"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# training a machine learning model using the linear regression algorithm
model = LinearRegression()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# Features = [["time_study", "number_courses"]]
features = np.array([[4.508, 3]])
model.predict(features)

