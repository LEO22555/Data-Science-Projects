import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Financial Budget Analysis\India_budget_2021.csv")
data.head()
print(data)
data.dropna()

# preparing the data by only selecting the main departments and putting all the other departments in the other category
data = data.iloc[[0,8,11,14,18,23,41,42,43],:]
row = {'Department /Ministry': 'OTHERS', 'Fund allotted(in ₹crores)': 592971.0800000001}
data = data.append(row, ignore_index = True)
print(data)

# look at the priorities of the government for the financial year
data.plot.bar(x='Department /Ministry', y='Fund allotted(in ₹crores)')

# plot this data into a donut plot to have a clear view of the distribution of funds among all the departments
df = data["Fund allotted(in ₹crores)"]
labels = data["Department /Ministry"]
plt.figure(figsize=(7,7))
plt.pie(df, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, shadow =True)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Distribution of The Budget", fontsize=20)
plt.show()

print("finance department is getting 40% of the funds")