import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\USUARIO\OneDrive - Universidad Nacional de Colombia\Desktop\DS\US President Heights\presidents.csv")
print(data.head())

height = np.array(data["height(cm)"])
print(height)

# summary statistics
print("Mean of heights =", height.mean())
print("Standard Deviation of height =", height.std())
print("Minimum height =", height.min())
print("Maximum height =", height.max())

# computing quantiles
print("25th percentile =", np.percentile(height, 25))
print("Median =", np.median(height))
print("75th percentile =", np.percentile(height, 75))

# Height Distribution of Presidents of USA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.hist(height)
plt.title("Height Distribution of Presidents of USA")
plt.xlabel("height(cm)")
plt.ylabel("Number")
plt.show()