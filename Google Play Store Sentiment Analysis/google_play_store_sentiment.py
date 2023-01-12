from itertools import count
from nltk.util import pr
import pandas as pd
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Google Play Store Sentiment Analysis\user_reviews.csv")
print(data.head())

print(data.isnull().sum())
data = data.dropna()
print(data.isnull().sum())

# adding three new columns in the dataset by understanding the sentiments of each customer review as Positive, Negative, and Neutral
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Translated_Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Translated_Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Translated_Review"]]
print(data.head())

# ook at the sentiments of customers about the applications available at the Google play store by using a scatter plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 10))
sns.scatterplot(data['Sentiment_Polarity'], data['Sentiment_Subjectivity'], 
                hue = data['Sentiment'], edgecolor='white', palette="twilight_shifted_r")
plt.title("Google Play Store Reviews Sentiment Analysis", fontsize=20)
plt.show()