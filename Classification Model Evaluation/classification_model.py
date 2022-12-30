import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Youtube01-Psy.csv")
data = data[["CONTENT", "CLASS"]]
x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

# Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(ytest, predictions)
print(confusionMatrix)

# Accuracy
print(model.score(xtest, ytest))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(ytest, predictions))

# AUC and ROC
import matplotlib.pyplot as plt
from sklearn import metrics
auc = metrics.roc_auc_score(ytest, predictions)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(ytest, predictions)

plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Using model evaluation metrics to evaluate the performance of your machine learning model