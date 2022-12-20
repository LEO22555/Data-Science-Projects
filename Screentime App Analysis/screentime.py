import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Screentime App Analysis\Screentime-App-Details.csv")
print(data.head())
data.isnull().sum()
print(data.describe())

# amount of usage of the apps
figure = px.bar(data_frame=data, x = "Date", y = "Usage", color="App", title="Usage")
figure.show()

# number of notifications from the apps
figure = px.bar(data_frame=data, x = "Date", y = "Notifications", color="App", title="Notifications")
figure.show()

#  number of times the apps opened
figure = px.bar(data_frame=data, x = "Date", y = "Times opened", color="App", title="Times Opened")
figure.show()

# relationship between the number of notifications and the amount of usage
figure = px.scatter(data_frame = data, x="Notifications",y="Usage", size="Notifications", trendline="ols", title = "Relationship Between Number of Notifications and Usage")
figure.show()

# There’s a linear relationship between the number of notifications and the amount of usage. It means that more notifications result in more use of smartphones