import pandas as pd
data = pd.read_csv(r'C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Click-Through Rate Prediction\Advertising.csv')
print(data.head())

print(data.isnull().sum())
print(data.columns)

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
x
y=data.iloc[:,9]
y

# using 70 per cent of data as training and 30 per cent as testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#  using the logistic regression model to predict the click-through rate of the users
from sklearn.linear_model import LogisticRegression
Lr=LogisticRegression(C=0.01,random_state=0)
Lr.fit(x_train,y_train)
y_pred=Lr.predict(x_test)
print(y_pred)

y_pred_proba=Lr.predict_proba(x_test)
print(y_pred_proba)

# looking at the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#  f1 score
from sklearn.metrics import f1_score
print(f1_score(y_test,y_pred))