import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv(r"Acelerometer Analysis\Acc Data\accdata.csv")
print(data.head())

# line plot with time on the x-axis and accelerometer data on the y-axis
fig = px.line(data, x="Date", 
              y=["accel_x", "accel_y", "accel_z"], 
              title="Acceleration data over time")
fig.show()

# average acceleration values by the hour of day and day of the week
data["hour"] = pd.to_datetime(data["Time"]).dt.hour
data["day_of_week"] = pd.to_datetime(data["Date"]).dt.day_name()
agg_data = data.pivot_table(index="hour", columns="day_of_week", 
                            values=["accel_x", "accel_y", "accel_z"], 
                            aggfunc="mean")

# Create a heatmap
fig = go.Figure(go.Heatmap(x=agg_data.columns.levels[1], 
                           y=agg_data.index, 
                           z=agg_data.values,
                           xgap=1, ygap=1, 
                           colorscale="RdBu", 
                           colorbar=dict(title="Average Acceleration")))
fig.update_layout(title="Average Acceleration by Hour of Day and Day of Week")
fig.show()

# representing the magnitude of the acceleration vector:
data['accel_mag'] = (data['accel_x'] ** 2 + data['accel_y'] ** 2 + data['accel_z'] ** 2) ** 0.5

# create a scatter plot of the magnitude of acceleration over time
fig = px.scatter(data, x='Time', 
                 y='accel_mag', 
                 title='Magnitude of Acceleration over time')
fig.show()

# 3D scatter plot where the x, y, and z axes represent the acceleration in each respective direction
fig = px.scatter_3d(data, x='accel_x', 
                    y='accel_y', 
                    z='accel_z', 
                    title='Acceleration in 3D space')
fig.show()

#  histogram to visualize the distribution of the magnitude of acceleration
fig = px.histogram(data, 
                   x='accel_mag', 
                   nbins=50, title='Acceleration magnitude histogram')
fig.show()