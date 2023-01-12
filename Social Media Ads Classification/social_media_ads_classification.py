import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Social Media Ads Classification\social.csv")
print(data.head())

print(data.describe())
print(data.isnull().sum())

# exploring some of the important patterns in the dataset
plt.figure(figsize=(15, 10))
plt.title("Product Purchased By People Through Social Media Marketing")
sns.histplot(x="Age", hue="Purchased", data=data)
plt.show()

plt.title("Product Purchased By People According to Their Income")
sns.histplot(x="EstimatedSalary", hue="Purchased", data=data)
plt.show()

print("people with a monthly income of over 90,000 among the target audience are more interested in purchasing the product")

x = np.array(data[["Age", "EstimatedSalary"]])
y = np.array(data[["Purchased"]])

# splitting the data and train a social media ads classification model using the decision tree classifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

# classification report of the model
print(classification_report(ytest, predictions))