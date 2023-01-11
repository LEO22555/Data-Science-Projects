from rake_nltk import Rake
rake_nltk_var = Rake()

# setting a text to extract keywords
text = """ I am a programmer from Colombia, and I am here to guide you with Data Science, Machine Learning, Python, and C++ for free. I hope you will learn a lot in your journey towards Coding, Machine Learning and Artificial Intelligence with me."""

# extract the keywords from the text and print the output
rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()
print(keyword_extracted)