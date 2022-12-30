import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Password Strength Checker\data.csv", error_bad_lines=False)
print(data.head())

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})
print(data.sample(5))

# tokenizing and spliting the data into training and test sets
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  
x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)

# training a classification model to predict the strength of the password
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# checking the strength of a password using the trained model
import getpass
user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model.predict(data)
print(output)

# A password strength checker works by understanding the combination of digits, letters, and special symbols you use in your password.