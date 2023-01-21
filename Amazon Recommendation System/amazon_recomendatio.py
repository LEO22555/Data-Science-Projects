import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Amazon Recommendation System\ratings_Electronics.csv")
print(data.head())

data.columns = ['user_id', 'product_id','ratings','timestamp']

# selecting a sample
df = data[:int(len(data) * .1)]

# prepare the dataset for creating a recommendation system
counts = df['user_id'].value_counts()
data = df[df['user_id'].isin(counts[counts >= 50].index)]
data.groupby('product_id')['ratings'].mean().sort_values(ascending=False) 
final_ratings = data.pivot(index = 'user_id', columns ='product_id', values = 'ratings').fillna(0)

num_of_ratings = np.count_nonzero(final_ratings)
possible_ratings = final_ratings.shape[0] * final_ratings.shape[1]
density = (num_of_ratings/possible_ratings)
density *= 100
final_ratings_T = final_ratings.transpose()

grouped = data.groupby('product_id').agg({'user_id': 'count'}).reset_index()
grouped.rename(columns = {'user_id': 'score'},inplace=True)
training_data = grouped.sort_values(['score', 'product_id'], ascending = [0,1]) 
training_data['Rank'] = training_data['score'].rank(ascending=0, method='first') 
recommendations = training_data.head()

# function to generate recommendations based on the score of the product reviews
def recommend(id):     
    recommend_products = recommendations 
    recommend_products['user_id'] = id 
    column = recommend_products.columns.tolist() 
    column = column[-1:] + column[:-1] 
    recommend_products = recommend_products[column] 
    return recommend_products 

print(recommend(11))