from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

credit_scores = [650, 720, 580, 800, 690, 750, 600, 670, 710, 680]
loan_approval = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0] 

data = {"Credit_Score": credit_scores, "Loan_Approval": loan_approval}
df = pd.DataFrame(data)
print(df.head())

# training a Machine Learning model using the Decision Tree algorithm:

X = df[['Credit_Score']]
y = df['Loan_Approval']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# visualizing the decision-making process of the decision tree algorithm

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, rounded=True, 
          feature_names=['Credit Score'], 
          class_names=['Denied', 'Approved'])
plt.show()

# making predictions using the Decision Tree algorithm
user_credit_score = float(input("Enter your credit score: "))

prediction = clf.predict([[user_credit_score]])

if prediction[0] == 1:
    print("Congratulations! Your loan application is likely to be approved.")
else:
    print("We regret to inform you that your loan application is likely to be denied.")