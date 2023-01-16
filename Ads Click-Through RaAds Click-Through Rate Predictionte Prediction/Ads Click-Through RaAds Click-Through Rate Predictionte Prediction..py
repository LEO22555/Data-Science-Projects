import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_white"

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\ctr\ad_10000records.csv")
print(data.head())

# The "Clicked on Ad" column contains 0 and 1 values, where 0 means not clicked, and 1 means clicked. I'll transform these values into "yes" and "no"
data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 
                               1: "Yes"})

# analyzing the click-through rate based on the time spent by the users on the website
fig = px.box(data, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# analyzing the click-through rate based on the income of the users
fig = px.box(data, 
             x="Area Income",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Income", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# calculating the overall Ads click-through rate
# seeing the distribution of users

data["Clicked on Ad"].value_counts()

# 4917 out of 10000 users clicked on the ads. Let's calculate the CTR
click_through_rate = 4917 / 10000 * 100
print(click_through_rate)

# CTR is 49.17

# training a Machine Learning model to predict click-through rate
data["Gender"] = data["Gender"].map({"Male": 1, 
                               "Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=4)

# training the model using the random forecast classification algorithm

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# accuracy of the model

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

# 0.9615

# testing the model by making predictions

print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))