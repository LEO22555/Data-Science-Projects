import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Water quality\water_potability.csv")
data.head()

data = data.dropna()
data.isnull().sum()

# seeing the distribution of 0 and 1 in the Potability column
plt.figure(figsize=(15, 10))
sns.countplot(data.Potability)
plt.title("Distribution of Unsafe and Safe Water")
plt.show()

# Factors Affecting Water Quality: PH
import plotly.express as px
data = data
figure = px.histogram(data, x = "ph", color = "Potability", title= "Factors Affecting Water Quality: PH")
figure.show()

# Factors Affecting Water Quality: Hardness
figure = px.histogram(data, x = "Hardness", color = "Potability", title= "Factors Affecting Water Quality: Hardness")
figure.show()

# Factors Affecting Water Quality: Solids
figure = px.histogram(data, x = "Solids", color = "Potability", title= "Factors Affecting Water Quality: Solids")
figure.show()

# Factors Affecting Water Quality: Chloramines
figure = px.histogram(data, x = "Chloramines", color = "Potability", title= "Factors Affecting Water Quality: Chloramines")
figure.show()

# factor affecting water quality
figure = px.histogram(data, x = "Sulfate", color = "Potability", title= "Factors Affecting Water Quality: Sulfate")
figure.show()

# Factors Affecting Water Quality: Conductivity
figure = px.histogram(data, x = "Conductivity", color = "Potability", title= "Factors Affecting Water Quality: Conductivity")
figure.show()

# Factors Affecting Water Quality: Organic Carbon
figure = px.histogram(data, x = "Organic_carbon", color = "Potability", title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()

# Factors Affecting Water Quality: Trihalomethanes
figure = px.histogram(data, x = "Trihalomethanes", color = "Potability", title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()

# Factors Affecting Water Quality: Turbidity
figure = px.histogram(data, x = "Turbidity", color = "Potability", title= "Factors Affecting Water Quality: Turbidity")
figure.show()

# look at the correlation of all the features with respect to the Potability column in the dataset
correlation = data.corr()
correlation["ph"].sort_values(ascending=False)

# seeing which machine learning algorithm is best for this dataset by using the PyCaret library in Python
from pycaret.classification import *
clf = setup(data, target = "Potability", silent = True, session_id = 786)
compare_models()

# training the model and examine its predictions
model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()

print("results are looking satisfactory jeje 2/jan/23")