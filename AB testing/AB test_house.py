import pandas as pd
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

control_data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\AB testing\control_group.csv", sep = ";")
test_data = pd.read_csv(r"C:\Users\USUARIO\Desktop\DS\AB testing\test_group.csv", sep = ";")

print(control_data.head())
print(test_data.head())

control_data.columns = ["Campaign Name", "Date", "Amount Spent", "Number of Impressions", "Reach", "Website Clicks", "Searches Received", "Content Viewed", "Added to Cart","Purchases"]
test_data.columns = ["Campaign Name", "Date", "Amount Spent", "Number of Impressions", "Reach", "Website Clicks", "Searches Received", "Content Viewed", "Added to Cart","Purchases"]

print(control_data.isnull().sum())
print(test_data.isnull().sum())

control_data["Number of Impressions"].fillna(value=control_data["Number of Impressions"].mean(), inplace=True)
control_data["Reach"].fillna(value=control_data["Reach"].mean(), inplace=True)
control_data["Website Clicks"].fillna(value=control_data["Website Clicks"].mean(), inplace=True)
control_data["Searches Received"].fillna(value=control_data["Searches Received"].mean(), inplace=True)
control_data["Content Viewed"].fillna(value=control_data["Content Viewed"].mean(), inplace=True)
control_data["Added to Cart"].fillna(value=control_data["Added to Cart"].mean(), inplace=True)
control_data["Purchases"].fillna(value=control_data["Purchases"].mean(), inplace=True)

# merge both db and reset index
ab_data = control_data.merge(test_data, how="outer").sort_values(["Date"])
ab_data = ab_data.reset_index(drop=True)
print(ab_data.head())
print(ab_data["Campaign Name"].value_counts())

# Scatter plot
figure = px.scatter(data_frame = ab_data, x="Number of Impressions", y="Amount Spent", size="Amount Spent", color= "Campaign Name", trendline="ols")
figure.show()


# Control Vs Test: Searches
label = ["Total Searches from Control Campaign", "Total Searches from Test Campaign"]
counts = [sum(control_data["Searches Received"]), sum(test_data["Searches Received"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Searches')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Control Vs Test: Website Clicks
label = ["Website Clicks from Control Campaign", "Website Clicks from Test Campaign"]
counts = [sum(control_data["Website Clicks"]), sum(test_data["Website Clicks"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Website Clicks')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Control Vs Test: Content Viewed
label = ["Content Viewed from Control Campaign", "Content Viewed from Test Campaign"]
counts = [sum(control_data["Content Viewed"]), sum(test_data["Content Viewed"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Content Viewed')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Control Vs Test: Added to Cart
label = ["Products Added to Cart from Control Campaign", "Products Added to Cart from Test Campaign"]
counts = [sum(control_data["Added to Cart"]), 
sum(test_data["Added to Cart"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Added to Cart')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Control Vs Test: Amount Spent
label = ["Amount Spent in Control Campaign", "Amount Spent in Test Campaign"]
counts = [sum(control_data["Amount Spent"]), sum(test_data["Amount Spent"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Amount Spent')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# Control Vs Test: Purchases
label = ["Purchases Made by Control Campaign", "Purchases Made by Test Campaign"]
counts = [sum(control_data["Purchases"]), sum(test_data["Purchases"])]
colors = ['gold','lightgreen']
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Control Vs Test: Purchases')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

# relationship between the number of website clicks and content viewed from both campaigns
figure = px.scatter(data_frame = ab_data, x="Content Viewed", y="Website Clicks", size="Website Clicks", color= "Campaign Name", trendline="ols")
figure.show()

# relationship between the amount of content viewed and the number of products added to the cart from both campaigns
figure = px.scatter(data_frame = ab_data, x="Added to Cart", y="Content Viewed", size="Added to Cart", color= "Campaign Name", trendline="ols")
figure.show()

# relationship between the number of products added to the cart and the number of sales from both campaigns
figure = px.scatter(data_frame = ab_data, x="Purchases", y="Added to Cart", size="Purchases", color= "Campaign Name", trendline="ols")
figure.show()

#  The control campaign resulted in more sales and engagement from the visitors. More products were viewed from the control campaign, resulting in more products in the cart and more sales. But the conversation rate of products in the cart is higher in the test campaign. The test campaign resulted in more sales according to the products viewed and added to the cart. And the control campaign results in more sales overall. So, the Test campaign can be used to market a specific product to a specific audience, and the Control campaign can be used to market multiple products to a wider audience.