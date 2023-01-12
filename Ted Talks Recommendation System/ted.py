import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Ted Talks Recommendation System\ted_talks.csv")
print(data.head())

# creating a new column as a title by separating the title from the URL
data["title"] = data["url"].map(lambda x:x.split("/")[-1])

# preparing this dataset and then let’s use cosine similarity to measure the similarities between different Ted talks
from sklearn.feature_extraction import text
ted_talks = data["transcript"].tolist()
bi_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words="english", ngram_range=(1,2))
bi_matrix = bi_tfidf.fit_transform(ted_talks)

uni_tfidf = text.TfidfVectorizer(input=ted_talks, stop_words="english")
uni_matrix = uni_tfidf.fit_transform(ted_talks)

from sklearn.metrics.pairwise import cosine_similarity
bi_sim = cosine_similarity(bi_matrix)
uni_sim = cosine_similarity(uni_matrix)

#  create a Python function to recommend ted talks based on their content
def recommend_ted_talks(x):
    return ".".join(data["title"].loc[x.argsort()[-5:-1]])
    
data["ted_talks_uni"] = [recommend_ted_talks(x) for x in uni_sim]
data["ted_talks_bi"] = [recommend_ted_talks(x) for x in bi_sim]
print(data['ted_talks_uni'].str.replace("_", " ").str.upper().str.strip().str.split("\n")[1])