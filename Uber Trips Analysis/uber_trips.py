
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\Uber Trips Analysis\uber-raw-data-sep14.csv")
data["Date/Time"] = data["Date/Time"].map(pd.to_datetime) 
data.head()

# Analyzing the Uber trips according to days and hours
data["Day"] = data["Date/Time"].apply(lambda x: x.day)
data["Weekday"] = data["Date/Time"].apply(lambda x: x.weekday())
data["Hour"] = data["Date/Time"].apply(lambda x: x.hour)
print(data.head())

# using the Uber trips for the September month so let’s have a look at each day to see on which day the Uber trips were highest
sns.set(rc={'figure.figsize':(12, 10)})
sns.distplot(data["Day"])

# Uber trips according to the hours
sns.distplot(data["Hour"])

# analyzing the Uber trips according to the weekdays
sns.distplot(data["Weekday"])

#  correlation of hours and weekdays on the Uber trips
df = data.groupby(["Weekday", "Hour"]).apply(lambda x: len(x))
df = df.unstack()
sns.heatmap(df, annot=False)

# plot the density of Uber trips according to the regions of the New Your city
data.plot(kind='scatter', x='Lon', y='Lat', alpha=0.4, s=data['Day'], label='Uber Trips',
figsize=(12, 8), cmap=plt.get_cmap('jet'))
plt.title("Uber Trips Analysis")
plt.legend()
plt.show()

# from this analysis
# Monday is the most profitable day for Uber
# On Saturdays less number of people use Uber
# 6 pm is the busiest day for Uber
# On average a rise in Uber trips start around 5 am.
# Most of the Uber trips originate near the Manhattan region in New York.