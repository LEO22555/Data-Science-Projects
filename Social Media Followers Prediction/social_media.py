import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Social Media Followers Prediction\stats.csv")
data.drop(data.tail(1).index, inplace=True)
data.head()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Number of Followers Gained Every Month")
sns.barplot(x="followers_gained", y="period_end", data=data)
plt.show()

# Total number of followers for each month
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Followers At The End of Every Month")
sns.barplot(x="followers_total", y="period_end", data=data)
plt.show()

# total number of views to each month
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Views Every Month")
sns.barplot(x="views", y="period_end", data=data)
plt.show()

# predicting the increase in the number of followers we can expect to see over the next four months
from autots import AutoTS
model = AutoTS(forecast_length=4, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='period_end', value_col='followers_gained', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)