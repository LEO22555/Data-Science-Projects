import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Smartwatch Data Analysis\dailyActivity_merged.csv")
print(data.head())

print(data.isnull().sum())

print(data.info())

# Changing datatype of ActivityDate
data["ActivityDate"] = pd.to_datetime(data["ActivityDate"], format="%m/%d/%Y")
print(data.info())

# combining all these columns as total minutes before moving forward
data["TotalMinutes"] = data["VeryActiveMinutes"] + data["FairlyActiveMinutes"] + data["LightlyActiveMinutes"] + data["SedentaryMinutes"]
print(data["TotalMinutes"].sample(5))

print(data.describe())

# Relationship between Calories & Total Steps
figure = px.scatter(data_frame = data, x="Calories",y="TotalSteps", size="VeryActiveMinutes", trendline="ols", title="Relationship between Calories & Total Steps")
figure.show()

label = ["Very Active Minutes", "Fairly Active Minutes", 
         "Lightly Active Minutes", "Inactive Minutes"]
counts = data[["VeryActiveMinutes", "FairlyActiveMinutes", 
               "LightlyActiveMinutes", "SedentaryMinutes"]].mean()
colors = ['gold','lightgreen', "pink", "blue"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Total Active Minutes')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

#  adding a new column to this dataset as “Day”
data["Day"] = data["ActivityDate"].dt.day_name()
print(data["Day"].head())

# ooking at the very active, fairly active, and lightly active minutes on each day of the week
fig = go.Figure()
fig.add_trace(go.Bar(x=data["Day"], y=data["VeryActiveMinutes"], name='Very Active', marker_color='purple'
))
fig.add_trace(go.Bar(x=data["Day"], y=data["FairlyActiveMinutes"], name='Fairly Active', marker_color='green'
))
fig.add_trace(go.Bar( x=data["Day"], y=data["LightlyActiveMinutes"], name='Lightly Active', marker_color='pink'
))
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

# look at the number of inactive minutes on each day of the week
day = data["Day"].value_counts()
label = day.index
counts = data["SedentaryMinutes"]
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Inactive Minutes Daily')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

calories = data["Day"].value_counts()
label = calories.index
counts = data["Calories"]
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

#  looking at the number of calories burned on each day of the week
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Calories Burned Daily')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

print("Tuesday is, therefore, one of the most active days for all individuals in the dataset, as the highest number of calories were burned on Tuesdays.")