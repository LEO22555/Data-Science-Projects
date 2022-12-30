import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Virat Kohli Performance Analysis\Virat_Kohli.csv")
print(data.head())

print(data.isnull().sum())

# Total Runs Between 18-Aug-08 - 22-Jan-17
data["Runs"].sum()

# Average Runs Between 18-Aug-08 - 22-Jan-17
data["Runs"].mean()

# trend of runs scored by Virat Kohli in his career from 18 August 2008 to 22 January 2017
matches = data.index
figure = px.line(data, x=matches, y="Runs", title='Runs Scored by Virat Kohli Between 18-Aug-08 - 22-Jan-17')
figure.show()

# Batting Positions
data["Pos"] = data["Pos"].map({3.0: "Batting At 3", 4.0: "Batting At 4", 2.0: "Batting At 2", 1.0: "Batting At 1", 7.0:"Batting At 7", 5.0:"Batting At 5", 6.0: "batting At 6"})

Pos = data["Pos"].value_counts()
label = Pos.index
counts = Pos.values
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Number of Matches At Different Batting Positions')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

label = data["Pos"]
counts = data["Runs"]
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

# total runs scored by Virat Kohli in different positions
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Runs By Virat Kohli At Different Batting Positions')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Centuries By Virat Kohli in First Innings Vs. Second Innings
centuries = data.query("Runs >= 100")
figure = px.bar(centuries, x=centuries["Inns"], y = centuries["Runs"], color = centuries["Runs"], title="Centuries By Virat Kohli in First Innings Vs. Second Innings")
figure.show()

# Dismissals of Virat Kohli
dismissal = data["Dismissal"].value_counts()
label = dismissal.index
counts = dismissal.values
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Dismissals of Virat Kohli')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Most Runs Against Teams
figure = px.bar(data, x=data["Opposition"], y = data["Runs"], color = data["Runs"], title="Most Runs Against Teams")
figure.show()

# Most Centuries Against Teams
figure = px.bar(centuries, x=centuries["Opposition"], y = centuries["Runs"], color = centuries["Runs"], title="Most Centuries Against Teams")
figure.show()

# create a new dataset of all the matches played by Virat Kohli where his strike rate was more than 120
strike_rate = data.query("SR >= 120")
print(strike_rate)

# Virat Kohli's High Strike Rates in First Innings Vs. Second Innings
figure = px.bar(strike_rate, x = strike_rate["Inns"], y = strike_rate["SR"], color = strike_rate["SR"], title="Virat Kohli's High Strike Rates in First Innings Vs. Second Innings")
figure.show()

# Relationship Between Runs Scored and Fours
figure = px.scatter(data_frame = data, x="Runs", y="4s", size="SR", trendline="ols", title="Relationship Between Runs Scored and Fours")
figure.show()

# Relationship Between Runs Scored and Sixes
figure = px.scatter(data_frame = data, x="Runs", y="6s", size="SR", trendline="ols", title= "Relationship Between Runs Scored and Sixes")
figure.show()

# There is no strong linear relationship here. It means Virat Kohli likes playing fours more than sixes.