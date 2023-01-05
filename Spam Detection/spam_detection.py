
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Spam Detection\spam.csv", encoding= 'latin-1')
data.head()

data = data[["class", "message"]]

# splitting this dataset into training and test sets and train the model to detect spam messages
x = np.array(data["message"])
y = np.array(data["class"])
cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)

# test this model by taking a user input as a message to detect whether it is spam or not
sample = input('Enter a message:')
data = cv.transform([sample]).toarray()
print(clf.predict(data))