import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
data = pd.read_csv('Prod/dataset.csv', names=['time', 'voltage', 'conductivity'])

# display the first 5 rows of the data
print(data.head())

# check the data types and missing values
print(data.info())

# summary statistics
print(data.describe())

# plot the conductivity distribution
sns.histplot(data['conductivity'])
plt.xlabel('Conductivity')
plt.title('Conductivity Distribution')
plt.show()