import numpy as np

# Suppose we have the following data
data = np.array([1, 2, 5, 6, 9, 12, 18, 25, 30, 40, 55, 60, 120])

# Calculate Q1, Q3, and IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
print('Outliers: ', outliers)