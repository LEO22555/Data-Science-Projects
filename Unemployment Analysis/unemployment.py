import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv")
print(data.head())

print(data.isnull().sum())


# renaming all the columns
data.columns= ["States","Date","Frequency", "Estimated Unemployment Rate", "Estimated Employed", "Estimated Labour Participation Rate", "Region","longitude","latitude"]

# correlation between the features of this dataset
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()


# looking at the estimated number of employees according to different regions of India
data.columns= ["States","Date","Frequency", "Estimated Unemployment Rate","Estimated Employed", "Estimated Labour Participation Rate","Region", "longitude","latitude"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=data)
plt.show()

# unemployment rate according to different regions of India
plt.figure(figsize=(12, 10))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
plt.show()

# dashboard to analyze the unemployment rate of each Indian state by region
unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], values="Estimated Unemployment Rate", width=700, height=700, color_continuous_scale="RdY1Gn", title="Unemployment Rate in India")
figure.show()

# Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force