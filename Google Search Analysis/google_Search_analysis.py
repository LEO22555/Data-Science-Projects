import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
trends = TrendReq()

trends.build_payload(kw_list=["Machine Learning"])
data = trends.interest_by_region()
data = data.sort_values(by="Machine Learning", ascending=False)
data = data.head(10)
print(data)

# visualize this data using a bar chart
data.reset_index().plot(x="geoName", y="Machine Learning", figsize=(15,12), kind="bar")
plt.style.use('fivethirtyeight')
plt.show()

# rend of searches to see how the total search queries based on “Machine Learning”
data = TrendReq(hl='en-US', tz=360)
data.build_payload(kw_list=['Machine Learning'])
data = data.interest_over_time()
fig, ax = plt.subplots(figsize=(15, 12))
data['Machine Learning'].plot()
plt.style.use('fivethirtyeight')
plt.title('Total Google Searches for Machine Learning', fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Total Count')
plt.show()

# see a huge increase in the searches about “machine learning” on Google in 2022