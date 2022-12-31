import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
data = fetch_openml("mnist_784", version=1)
print(data)

x, y = data["data"], data["target"]
print(x.shape)

# The dataset contains 70,000 rows and 784 columns
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# look at the sample of the kind of images we have in the dataset
image = np.array(xtrain.iloc[0]).reshape(28, 28)
plt.imshow(image)

#  using the stochastic gradient descent classification algorithm
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(xtrain, ytrain)

# testing the trained model by making predictions on the test set
predictions = model.predict(xtest)
print(predictions)

# look at the handwritten digits images to evaluate our predictions
image = np.array(xtest.iloc[0]).reshape(28, 28)
plt.imshow(image)

image = np.array(xtest.iloc[1]).reshape(28, 28)
plt.imshow(image)


