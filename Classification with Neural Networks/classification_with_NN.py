import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

#  Quick look at one of the samples of the images from the dataset
imgIndex = 9
image = xtrain[imgIndex]
print("Image Label :",ytrain[imgIndex])
plt.imshow(image)

print(xtrain.shape)
print(xtest.shape)

# Building a neural network architecture with two hidden layers
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())

# spliting the training data into training and validation sets
xvalid, xtrain = xtrain[:5000]/255.0, xtrain[5000:]/255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

# train a neural network for the task of image classification
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=30, validation_data=(xvalid, yvalid))

# predictions
new = xtest[:5]
predictions = model.predict(new)
print(predictions)

# predicted classes
classes = np.argmax(predictions, axis=1)
print(classes)