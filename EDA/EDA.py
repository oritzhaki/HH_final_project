import pandas as pd

data = pd.read_csv('Prod/dataset.csv', header=None, names=['time', 'voltage', 'conductivity'])


print(data.shape)       # print the number of rows and columns in the dataframe
print(data.columns)     # print the names of the columns
print(data.dtypes)      # print the data types of the columns


print(data.isna().sum())   # print the number of missing values in each column


print(data.describe())   # calculate summary statistics for each numeric column


import matplotlib.pyplot as plt

VOLTS = [-90, -80, -70, -60, -50, -40, -30, -20, -10]

# plot the conductivity vs. time for each voltage level
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flatten()):
    df = data[data['voltage'] == VOLTS[i]]
    ax.plot(df['time'], df['conductivity'])
    ax.set_title(f'Voltage = {VOLTS[i]} mV')
plt.tight_layout()
plt.show()

# plot the mean conductivity vs. time for all voltage levels
mean_conductivity = data.groupby('time')['conductivity'].mean()
plt.plot(mean_conductivity.index, mean_conductivity)
plt.xlabel('Time (ms)')
plt.ylabel('Mean Conductivity')
plt.show()
