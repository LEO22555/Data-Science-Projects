import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Topic Modelling\articles.csv", encoding = 'latin1')
print(data.head())

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]
    # Join tokens to form preprocessed text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

data['Article'] = data['Article'].apply(preprocess_text)

# convert the textual data into a numerical representation
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data['Article'].values)

# use the LDA algorithm to assign topic labels
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(x)

topic_modelling = lda.transform(x)

topic_labels = np.argmax(topic_modelling, axis=1)
data['topic_labels'] = topic_labels

# print labeled results
print(data.head())