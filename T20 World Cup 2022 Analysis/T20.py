import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\T20 World Cup 2022 Analysis\t20-world-cup-22.csv")
print(data.head())

# Number of Matches Won by teams in t20 World Cup 2022
figure = px.bar(data, x=data["winner"],title="Number of Matches Won by teams in t20 World Cup 2022")
figure.show()

won_by = data["won by"].value_counts()
label = won_by.index
counts = won_by.values
colors = ['gold','lightblue']

# Number of Matches Won By Runs Or Wickets
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Number of Matches Won By Runs Or Wickets')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Toss Decisions in t20 World Cup 2022
toss = data["toss decision"].value_counts()
label = toss.index
counts = toss.values
colors = ['blue','yellow']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Toss Decisions in t20 World Cup 2022')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Top Scorers in t20 World Cup 2022
figure = px.bar(data, x=data["top scorer"], y = data["highest score"], color = data["highest score"], title="Top Scorers in t20 World Cup 2022")
figure.show()

# Player of the Match Awards in t20 World Cup 2022
figure = px.bar(data, x = data["player of the match"], title="Player of the Match Awards in t20 World Cup 2022")
figure.show()

# Best Bowlers in t20 World Cup 2022
figure = px.bar(data, x=data["best bowler"],title="Best Bowlers in t20 World Cup 2022")
figure.show()

# Best Stadiums to Bat First or Chase
fig = go.Figure()
fig.add_trace(go.Bar(x=data["venue"], y=data["first innings score"], name='First Innings Runs', marker_color='blue'))
fig.add_trace(go.Bar(x=data["venue"], y=data["second innings score"], name='Second Innings Runs', marker_color='red'))
fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Best Stadiums to Bat First or Chase")
fig.show()

# Best Statiums to Bowl First or Defend
fig = go.Figure()
fig.add_trace(go.Bar(x=data["venue"], y=data["first innings wickets"], name='First Innings Wickets', marker_color='blue'))
fig.add_trace(go.Bar(x=data["venue"], y=data["second innings wickets"], name='Second Innings Wickets', marker_color='red'))
fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Best Statiums to Bowl First or Defend")
fig.show()

# England won the most number of matches
# Virat Kohli scored highest in the most number of matches
# Sam Curran was the best bowler in the most number of matches
# More teams won by batting first
# More teams decided to bat first
# SCG was the best stadium to bat first
# SCG was the best stadium to defend the target in the World Cup
# The Optus Stadium was the best stadium to bowl first