import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_json(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Sarcasm Detection\Sarcasm.json", lines=True)
print(data.head())

# transforming the values of this column as “sarcastic” and “not sarcastic” instead of 1 and 0
data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
print(data.head())

#  preparing the data for training a machine learning model
data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#  using the Bernoulli Naive Bayes algorithm to train a model for the task of sarcasm detection
model = BernoulliNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)