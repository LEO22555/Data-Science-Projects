import pandas as pd
import numpy as np

data = {'A': [1, 2, 3, 4, np.nan, 6, 7, 8, 9, np.nan],
        'B': [2, 4, 6, 8, np.nan, 12, 14, 16, 18, np.nan],
        'C': ['red', 'blue', np.nan, 'green', 'green', 
              'blue', 'red', 'blue', 'green', np.nan]}
df = pd.DataFrame(data)
print(df)

# filling with mean
mean_A = df['A'].mean()
df['A'].fillna(mean_A, inplace=True)
print(df)

# filling with median
median_B = df['B'].median()
df['B'].fillna(median_B, inplace=True)
print(df)

# filling with mode
mode_C = df['C'].mode()[0]
df['C'].fillna(mode_C, inplace=True)
print(df)