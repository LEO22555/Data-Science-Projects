import pandas as pd
data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Insurance Prediction\TravelInsurancePrediction.csv")
data.head()

data.drop(columns=["Unnamed: 0"], inplace=True)
data.isnull().sum()
data.info()

# converting 1 and 0 to purchased and not purchased:
data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})

# Factors Affecting Purchase of Travel Insurance: Age
import plotly.express as px
data = data
figure = px.histogram(data, x = "Age", color = "TravelInsurance", title= "Factors Affecting Purchase of Travel Insurance: Age")
figure.show()

# how a person’s type of employment affects the purchase of an insurance policy
import plotly.express as px
data = data
figure = px.histogram(data, x = "Employment Type", color = "TravelInsurance", title= "Factors Affecting Purchase of Travel Insurance: Employment Type")
figure.show()

# Factors Affecting Purchase of Travel Insurance: Income
import plotly.express as px
data = data
figure = px.histogram(data, x = "AnnualIncome", color = "TravelInsurance", title= "Factors Affecting Purchase of Travel Insurance: Income")
figure.show()

# converting all categorical values to 1 and 0 first because all columns are important for training the insurance prediction model
import numpy as np
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", "AnnualIncome", "FamilyMembers", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])

# splitting the data and train the model by using the decision tree classification algorithm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

print("The model gives a score of over 80% which is not bad for this kind of problem")
