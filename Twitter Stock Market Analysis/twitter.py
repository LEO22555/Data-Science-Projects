import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Twitter Stock Market Analysis\TWTR.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

data = data.dropna()

# Twitter Stock Prices Over the Years
figure = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"],low=data["Low"], close=data["Close"])])
figure.update_layout(title = "Twitter Stock Prices Over the Years", xaxis_rangeslider_visible=False)
figure.show()

# chart to analyze the stock prices of Twitter in detail
figure = px.bar(data, x = "Date", y= "Close", color="Close")
figure.update_xaxes(rangeslider_visible=True)
figure.show()

figure = px.bar(data, x = "Date", y= "Close", color="Close")
figure.update_xaxes(rangeslider_visible=True)
figure.update_layout(title = "Twitter Stock Prices Over the Years", 
                     xaxis_rangeslider_visible=False)

# add buttons to analyze the stock prices of Twitter in different time periods
figure.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

figure.show()

# complete timeline of Twitter in the stock market
data["Date"] = pd.to_datetime(data["Date"], 
                              format = '%Y-%m-%d')
data['Year'] = data['Date'].dt.year
data["Month"] = data["Date"].dt.month
fig = px.line(data, 
              x="Month", 
              y="Close", 
              color='Year', 
              title="Complete Timeline of Twitter")
fig.show()

# Twitter in the stock market from 2013 to 2022. Twitter is a popular social media application and is still getting more popular after Elon Musk took over Twitter. But it never was among the best-performing companies in the stock market. I hope you liked this article on Twitter Stock Market Analysis using Python. Feel free to ask valuable questions in the comments section below.