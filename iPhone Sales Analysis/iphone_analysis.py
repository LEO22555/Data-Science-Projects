import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\iPhone Sales Analysis\apple_products.csv")
print(data.head())

print(data.isnull().sum())

print(data.describe())

# top 10 highest-rated iPhones in India on Flipkart
highest_rated = data.sort_values(by=["Star Rating"], ascending=False)
highest_rated = highest_rated.head(10)
print(highest_rated['Product Name'])

# Number of Ratings of Highest Rated iPhones
iphones = highest_rated["Product Name"].value_counts()
label = iphones.index
counts = highest_rated["Number Of Ratings"]
figure = px.bar(highest_rated, x=label, y = counts, title="Number of Ratings of Highest Rated iPhones")
figure.show()

# Number of Reviews of Highest Rated iPhones
iphones = highest_rated["Product Name"].value_counts()
label = iphones.index
counts = highest_rated["Number Of Reviews"]
figure = px.bar(highest_rated, x=label, y = counts, title="Number of Reviews of Highest Rated iPhones")
figure.show()

# Relationship between Sale Price and Number of Ratings of iPhones
figure = px.scatter(data_frame = data, x="Number Of Ratings", y="Sale Price", size="Discount Percentage", trendline="ols", title="Relationship between Sale Price and Number of Ratings of iPhones")
figure.show()

# Relationship between Discount Percentage and Number of Ratings of iPhones
figure = px.scatter(data_frame = data, x="Number Of Ratings", y="Discount Percentage", size="Sale Price", trendline="ols", title="Relationship between Discount Percentage and Number of Ratings of iPhones")
figure.show()

# There is a linear relationship between the discount percentage on iPhones on Flipkart and the number of ratings. It means iPhones with high discounts are sold more in India.
# APPLE iPhone 8 Plus (Gold, 64 GB) was the most appreciated iPhone in India
# iPhones with lower sale prices are sold more in India
# iPhones with high discounts are sold more in India

