import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Instagram Recommendation System\Instagram.csv")
print(data.head())

# choosing the caption and the hashtags column for the rest of the task
data = data[["Caption", "Hashtags"]]
print(data.head())

# Using consime cosine_similarity
captions = data["Caption"].tolist()
uni_tfidf = text.TfidfVectorizer(input=captions, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(captions)
uni_sim = cosine_similarity(uni_matrix)

def recommend_post(x):
  return ", ".join(data["Caption"].loc[x.argsort()[-5:-1]])

data["Recommended Post"] = [recommend_post(x) for x in uni_sim]
print(data.head())

print(data["Recommended Post"][3])