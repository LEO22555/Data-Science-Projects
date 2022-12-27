import pandas as pd
from datetime import date, timedelta
import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Business Forecasting\adidas-quarterly-sales.csv")
print(data)

# quarterly sales revenue of Adidas
import plotly.express as px
figure = px.line(data, x="Time Period", y="Revenue", title='Quarterly Sales Revenue of Adidas in Millions')
figure.show()

# seasonality of any time series data
result = seasonal_decompose(data["Revenue"], model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(15, 10)

#Using the Seasonal ARIMA (SARIMA) model to forecast the quarterly sales revenue of Adidas
pd.plotting.autocorrelation_plot(data["Revenue"])
plot_pacf(data["Revenue"], lags = 20)

model=sm.tsa.statespace.SARIMAX(data['Revenue'],order=(p, d, q),seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

#Forecasting the quarterly revenue of Adidas for the next eight quarters
predictions = model.predict(len(data), len(data)+7)
print(predictions)

#Plotting the predictions
data["Revenue"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")