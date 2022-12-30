import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\IPL 2022 Analysis\Book_ipl22_ver_33.csv")
print(data.head())

# Number of Matches Won in IPL 2022
figure = px.bar(data, x=data["match_winner"], title="Number of Matches Won in IPL 2022")
figure.show()

#  how most of the teams win. Analyzing whether most of the teams win by defending (batting first) or chasing (batting second)
data["won_by"] = data["won_by"].map({"Wickets": "Chasing", "Runs": "Defending"})
won_by = data["won_by"].value_counts()
label = won_by.index
counts = won_by.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Number of Matches Won By Defending Or Chasing')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# what most teams prefer (batting or fielding) after winning the toss
toss = data["toss_decision"].value_counts()
label = toss.index
counts = toss.values
colors = ['skyblue','yellow']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Toss Decision')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Top scorers of most IPL 2022 matches
figure = px.bar(data, x=data["top_scorer"], title="Top Scorers in IPL 2022")
figure.show()

figure = px.bar(data, x=data["top_scorer"], y = data["highscore"],  color = data["highscore"], title="Top Scorers in IPL 2022")
figure.show()

# Most Player of the Match Awards
figure = px.bar(data, x = data["player_of_the_match"], title="Most Player of the Match Awards")
figure.show()

# Best Bowlers in IPL 2022
figure = px.bar(data, x=data["best_bowling"], title="Best Bowlers in IPL 2022")
figure.show()

# whether most of the wickets fall while setting the target or while chasing the target
figure = go.Figure()
figure.add_trace(go.Bar(
    x=data["venue"],
    y=data["first_ings_wkts"],
    name='First Innings Wickets',
    marker_color='gold'
))
figure.add_trace(go.Bar(
    x=data["venue"],
    y=data["second_ings_wkts"],
    name='Second Innings Wickets',
    marker_color='lightgreen'
))
figure.update_layout(barmode='group', xaxis_tickangle=-45)
figure.show()

print("So in the Wankhede Stadium in Mumbai and MCA Stadium in Pune, most wickets fall while chasing the target. And in the other two stadiums, most wickets fall while setting the target. ")