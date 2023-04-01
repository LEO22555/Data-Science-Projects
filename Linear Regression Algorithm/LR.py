import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

#sample data
num_cupcakes = [10, 20, 30, 40, 50]
money_made = [25, 50, 75, 100, 125]

# Relationship between the number of cupcakes sold and the amount of money made

figure = go.Figure()
figure.add_trace(go.Scatter(x=num_cupcakes, y=money_made, mode='markers',))
figure.update_layout(title='Relationship Between Cupcake Sales and Money Made',
                  xaxis_title='Number of Cupcakes Sold (Independent Variable)',
                  yaxis_title='Money Made (Dependent Variable)')
figure.show()

# Training a Machine Learning model using the linear regression algorithm:

#creating a linear regression model
x = np.array(num_cupcakes).reshape((-1, 1))
y = np.array(money_made)
model = LinearRegression().fit(x, y)

# predicting the value of a dependent variable using the value of an independent variable using our Machine Learning model

#predict the money made for 25 cupcakes sold
new_num_cupcakes = [[25]]
new_money_made = model.predict(new_num_cupcakes)
print("Predicted money made for the cupcakes sold:", new_money_made[0])