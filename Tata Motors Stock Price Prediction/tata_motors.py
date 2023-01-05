import numpy as np
import pandas as pd
import plotly.graph_objects as go
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS2\Tata Motors Stock Price Prediction\TTM.csv")
print(data.head())

# interactive visualisation of the stock prices to get a clear picture of the increase and decrease of the stock prices of Tata Motors
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], high=data["High"],
                                        low=data["Low"], close=data["Close"])])
figure.update_layout(title = "Tata Motors Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()

print(data.corr())

# predict the stock prices of Tata Motors
from autots import AutoTS
model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)