import numpy as np
import pandas as pd
movies = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Movie Rating Analysis\movies.dat", delimiter='::')
print(movies.head())

movies.columns = ["ID", "Title", "Genre"]
print(movies.head())

# rating dataset
ratings = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Movie Rating Analysis\ratings.dat", delimiter='::')
print(ratings.head())

# defining the column names of this data 
ratings.columns = ["User", "ID", "Ratings", "Timestamp"]
print(ratings.head())

# merging these two datasets into one
data = pd.merge(movies, ratings, on=["ID", "ID"])
print(data.head())

# distribution of the ratings of all the movies given by the viewers
ratings = data["Ratings"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(data, values=quantity, names=numbers)
fig.show()

# take a look at the top 10 movies that got 10 ratings by viewers
data2 = data.query("Ratings == 10")
print(data2["Title"].value_counts().head(10))

print("according to this dataset, Joker (2019) got the highest number of 10 ratings from viewers")
