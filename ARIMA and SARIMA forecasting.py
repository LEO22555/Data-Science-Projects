# ARIMA stands for Autoregressive Integrated Moving Average. 
# It is an algorithm used for forecasting Time Series Data. 
# ARIMA models have three parameters like ARIMA(p, d, q). Here p, d, and q are defined as:

# p is the number of lagged values that need to be added or subtracted from the values (label column). 
# It captures the autoregressive part of ARIMA.
# 
# d represents the number of times the data needs to differentiate to produce a stationary signal. 
# If it’s stationary data, the value of d should be 0, and if it’s seasonal data, the value of d should be 1.
# d captures the integrated part of ARIMA.

# q is the number of lagged values for the error term added or subtracted from the values (label column). 
# It captures the moving average part of ARIMA.

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('GOOG', start=start_date, end=end_date, progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
print(data.tail())

# Visualize data
data = data[["Date", "Close"]]
print(data.head())

# visualize the close prices of Google before moving forward
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Close"])


#  seasonal decomposition method that splits the time series data into trend, seasonal, and residuals for a better understanding of the time series data
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data["Close"], model='multiplicative', period = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 10)

# finding p value of p
pd.plotting.autocorrelation_plot(data["Close"])

# ind the value of q (moving average)
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Close"], lags = 100)

# building the ARIMA model:
p, d, q = 5, 1, 2
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(data["Close"], order=(p,d,q))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# predicting the values using the ARIMA model

predictions = fitted.predict()
print(predictions)

# building the SARIMA model:
import statsmodels.api as sm
import warnings
model=sm.tsa.statespace.SARIMAX(data['Close'],order=(p, d, q),seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())


# predict the future stock prices using the SARIMA model for the next 10 days
predictions = model.predict(len(data), len(data)+10)
print(predictions)

# Plotting predictions
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")


# If the data is stationary, we need to use ARIMA, if the data is seasonal, we need to use Seasonal ARIMA (SARIMA). 