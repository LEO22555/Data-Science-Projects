import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=720)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
print(data.head())

# Time Series Analysis (Line Plot)
import plotly.express as px
figure = px.line(data, x = data.index, y = "Close", title = "Time Series Analysis (Line Plot)")
figure.show()


#  candlestick chart to see the trends in the open, high, low, and close prices of Apple
import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x = data.index, open = data["Open"], high = data["High"], low = data["Low"], close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart)", xaxis_rangeslider_visible = False)
figure.show()

# Visualizing a bar plot to visualize the trends of close prices over the period
figure = px.bar(data, x = data.index, y = "Close", title = "Time Series Analysis (Bar Plot)" )
figure.show()

# Time Series Analysis (Custom Date Range)
figure = px.line(data, x = data.index, y = 'Close', range_x = ['2021-07-01','2021-12-31'], title = "Time Series Analysis (Custom Date Range)")
figure.show()

# Creating an interactive candlestick chart
figure = go.Figure(data = [go.Candlestick(x = data.index,
                                        open = data["Open"], 
                                        high = data["High"],
                                        low = data["Low"], 
                                        close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart with Buttons and Slider)")

figure.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
            dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
            dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
            dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
            dict(step = "all")
        ])
    )
)
figure.show()