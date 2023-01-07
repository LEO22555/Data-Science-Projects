
import numpy as np # linear algebra
import pandas as pd # data processing

import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

tv_shows = pd.read_csv(r'C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Best Streaming Service Analysis\tv_shows.csv')
tv_shows.head()

tv_shows.drop_duplicates(subset='Title', keep='first',inplace=True)

# fill the null values in the data with zeroes and then convert them into integer data types
tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].fillna('0%')
tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].apply(lambda x : x.rstrip('%'))
# tv_shows['Rotten Tomatoes'] = pd.to_numeric(tv_shows['Rotten Tomatoes'])

tv_shows['IMDb'] = tv_shows['IMDb'].fillna(0)
tv_shows['IMDb'] = tv_shows['IMDb']*10
tv_shows['IMDb'] = tv_shows['IMDb'].astype('int')

#  1s and 0s in the columns named Netflix, Hulu, Disney and Prime Video under a categorical format
tv_shows_long=pd.melt(tv_shows[['Title','Netflix','Hulu','Disney+', 'Prime Video']],id_vars=['Title'], var_name='StreamingOn', value_name='Present')
tv_shows_long = tv_shows_long[tv_shows_long['Present'] == 1]
tv_shows_long.drop(columns=['Present'],inplace=True)


# merge this data with the data we started with but I will drop some unwanted columns
tv_shows_combined = tv_shows_long.merge(tv_shows, on='Title', how='inner')
tv_shows_combined.drop(columns = ['Unnamed: 0','Netflix', 'Hulu', 'Prime Video', 'Disney+', 'type'], inplace=True)

# plat the data where the rantings are more than 1 to see the quantity of the tv shows available on each platform
tv_shows_both_ratings = tv_shows_combined[(tv_shows_combined.IMDb > 0) & tv_shows_combined['Rotten Tomatoes'] > 0]
tv_shows_combined.groupby('StreamingOn').Title.count().plot(kind='bar')


# violin charts to gauge the content ratings and the freshness of the streaming platform
figure = []
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'IMDb', color='StreamingOn'))
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'Rotten Tomatoes', color='StreamingOn'))
fig = make_subplots(rows=2, cols=4, shared_yaxes=True)

for i in range(2):
    for j in range(4):
        fig.add_trace(figure[i]['data'][j], row=i+1, col=j+1)

fig.update_layout(autosize=False, width=800, height=800)        
fig.show()

# scatter plot to compare the ratings between IMBD and Rotten Tomatoes to compare which streaming platform has the best ratings in both the user rating platforms
px.scatter(tv_shows_both_ratings, x='IMDb', y='Rotten Tomatoes',color='StreamingOn')

# By using the violin chart we can observe that:

# Hulu, Netflix, and Amazon Videos all have important data. As content increases, quality decreases for all three.
# Prime Videos has become denser in the top half when looking at IMDB and performs well in cool.
# Disney+ being new, has also been very successful in this area.