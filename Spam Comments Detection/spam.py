import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Spam Comments Detection\Youtube01-Psy.csv")
print(data.sample(5))

# Selecting both the columns and move further
data = data[["CONTENT", "CLASS"]]
print(data.sample(5))

# The class column contains values 0 and 1. 0 indicates not spam, and 1 indicates spam
data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})
print(data.sample(5))

#  using the Bernoulli Naive Bayes algorithm to train the model
x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# test the model by giving spam and not spam comments as input
sample = "Check this out: https://leolagaver.com/" 
data = cv.transform([sample]).toarray()
print(model.predict(data))

sample = "Lack of information!" 
data = cv.transform([sample]).toarray()
print(model.predict(data)) 