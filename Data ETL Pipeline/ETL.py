import tensorflow.keras as keras
(xtrain, ytrain), (xtest, ytest) = keras.datasets.fashion_mnist.load_data()


# shape of the data
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#  cleaning and transforming the data. Normalizing the pixel values to be between 0 and 1 and reshape the data into a 4D tensor
import numpy as np

xtrain = xtrain.astype('float32') / 255
xtest = xtest.astype('float32') / 255

xtrain = np.reshape(xtrain, (xtrain.shape[0], 28, 28, 1))
xtest = np.reshape(xtest, (xtest.shape[0], 28, 28, 1))

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#  loading the data into a database (sqlite)
import sqlite3

conn = sqlite3.connect('fashion_mnist.db')

conn.execute('''CREATE TABLE IF NOT EXISTS images
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             image BLOB NOT NULL,
             label INTEGER NOT NULL);''')

for i in range(xtrain.shape[0]):
    conn.execute('INSERT INTO images (image, label) VALUES (?, ?)',
                [sqlite3.Binary(xtrain[i]), ytrain[i]])

conn.commit()

for i in range(xtest.shape[0]):
    conn.execute('INSERT INTO images (image, label) VALUES (?, ?)',
                [sqlite3.Binary(xtest[i]), ytest[i]])

conn.commit()

conn.close()

# reading the data you stored on the SQLite database

import sqlite3
conn = sqlite3.connect('fashion_mnist.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM images')
rows = cursor.fetchall()

import pandas as pd
data = pd.read_sql_query('SELECT * FROM images', conn)