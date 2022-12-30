import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\Book Recommendation System\book_data.csv")
print(data.head())

data = data[["book_title", "book_desc", "book_rating_count"]]
print(data.head())

data = data.sort_values(by="book_rating_count", ascending=False)
top_5 = data.head()

import plotly.express as px
import plotly.graph_objects as go

labels = top_5["book_title"]
values = top_5["book_rating_count"]
colors = ['gold','lightgreen']


# top 5 books in the dataset according to the number of ratings
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(title_text="Top 5 Rated Books")
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

print(data.isnull().sum())

data = data.dropna()

feature = data["book_desc"].tolist()
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['book_title']).drop_duplicates()

def book_recommendation(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:5]
    bookindices = [i[0] for i in similarity_scores]
    return data['book_title'].iloc[bookindices]

print(book_recommendation("Letters to a Secret Lover"))