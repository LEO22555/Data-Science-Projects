import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Website Traffic Forecasting\Thecleverprogrammer.csv")
print(data.head())

# converting the Date column into Datetime data type
data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
print(data.info())

# Daily traffic of the website
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Views"])
plt.title("Daily Traffic of Thecleverprogrammer.com")
plt.show()

# looking if whether our dataset is stationary or seasonal
result = seasonal_decompose(data["Views"], model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 10)

# As the data is not stationary, the value of d is 1. To find the values of p and q, we can use the autocorrelation and partial autocorrelation plots
pd.plotting.autocorrelation_plot(data["Views"])

plot_pacf(data["Views"], lags = 100)

# training a SARIMA model for the task of website traffic forecasting
p, d, q = 5, 1, 2
model=sm.tsa.statespace.SARIMAX(data['Views'], order=(p, d, q), seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

# forecasting traffic on the website for the next 50 days
predictions = model.predict(len(data), len(data)+50)
print(predictions)

# ploting the predictions
data["Views"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")