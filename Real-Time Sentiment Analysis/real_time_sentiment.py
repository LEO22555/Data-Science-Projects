from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# taking a user input and have a look at the sentiment score
user_input = input("Please Rate Our Services >>: ")
sid = SentimentIntensityAnalyzer()
score = sid.polarity_scores(user_input)
print(score)

user_input = input("Please Rate Our Services >>: ")
sid = SentimentIntensityAnalyzer()
score = sid.polarity_scores(user_input)
if score["neg"] != 0:
      print("Negative")
else:
      print("Positive")