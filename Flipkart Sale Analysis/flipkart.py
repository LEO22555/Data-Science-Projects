import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Flipkart Sale Analysis\data.csv")
print(data.head())

# Creating a new discount column by calculating the discount offered by Flipkart on every smartphone
data["Discount"] = (data['original_price'] - data['offer_price']) / data['original_price'] * 100

# Top deals on smartphones offered by Flipkart on the sale
top_deals = data.sort_values(by="Discount", ascending=False)
deals = top_deals["name"][:15].value_counts()
label = deals.index
counts = top_deals["Discount"][:15].values
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Highest Discount Deals in the Flipkart Big Billion Days Sale')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Highest-rated smartphones on Flipkart on this sale
highest_rated = data.sort_values(by="rating", ascending=False)
deals = highest_rated["name"][:10].value_counts()
label = deals.index
counts = highest_rated["rating"][:10].values
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Highest Rated Discount Deals in the Flipkart Big Billion Days Sale')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Most expensive smartphone deals in the sale
most_expensive = data.sort_values(by="offer_price", ascending=False)
deals = most_expensive["name"][:10].value_counts()
label = deals.index
counts = most_expensive["offer_price"][:10].values
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Most Expensive Offers in the Flipkart Big Billion Days Sale')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Calculating the cost of this sale to Flipkart on just smartphones on the first day of the sale
label = ["Total of Offer Prices in Sales", "Total of Original Prices (MRP)"]
counts = [sum(data["offer_price"]), sum(data["original_price"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Total Discounts Offered Vs. MRP')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# The cost of a discount to Flipkart on just smartphones will be ₹23,53,772 for just one quantity of all the smartphones offered in the sale