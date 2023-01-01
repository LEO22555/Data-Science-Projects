import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Covid-19 Deaths Prediction\COVID19 data for overall INDIA.csv")
print(data.head())

data.isnull().sum()

data = data.drop("Date", axis=1)

import plotly.express as px
fig = px.bar(data, x='Date_YMD', y='Daily Confirmed')
fig.show()

# visualizing the death rate due to Covid-19
cases = data["Daily Confirmed"].sum()
deceased = data["Daily Deceased"].sum()

labels = ["Confirmed", "Deceased"]
values = [cases, deceased]

fig = px.pie(data, values=values, names=labels, title='Daily Confirmed Cases vs Daily Deaths', hole=0.5)
fig.show()

# calculating the death rate of Covid-19
death_rate = (data["Daily Deceased"].sum() / data["Daily Confirmed"].sum()) * 100
print(death_rate)

# daily deaths of covid-19
import plotly.express as px
fig = px.bar(data, x='Date_YMD', y='Daily Deceased')
fig.show()

# predict covid-19 deaths with machine learning for the next 30 days
from autots import AutoTS
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date_YMD", value_col='Daily Deceased', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)