import pandas as pd
import matplotlib.pyplot as plt

# Read the two dataframes
based_generated = pd.read_csv('GeneratedDataBasedGAParams/dataset_based_ga_params.csv', header=None)
based_generated = based_generated.drop(based_generated.index[0])
Test_df = pd.read_csv('DatasetTest/test.csv', header=None)
Test_df = Test_df.drop(Test_df.index[0])

based_generated.columns = ['Time', 'Temp', 'Value']
Test_df.columns = ['Time', 'Temp', 'Value']

# Convert 'Time' and 'Value' to numeric types
based_generated['Time'] = pd.to_numeric(based_generated['Time'])
based_generated['Value'] = pd.to_numeric(based_generated['Value'])
Test_df['Time'] = pd.to_numeric(Test_df['Time'])
Test_df['Value'] = pd.to_numeric(Test_df['Value'])

# Plot df1 with smaller markers
plt.plot(based_generated['Time'], based_generated['Value'], 'o', label='Based On GA Params', markersize=1)

# Plot df2 with smaller markers
plt.plot(Test_df['Time'], Test_df['Value'], 'o', label='Test (Truth + Noise)', markersize=1)

# Add title and labels
plt.title('Dataset Comparison')
plt.xlabel('Time (ms)')
plt.ylabel('Conductivity (G)')

# Adjust the legend position outside the plot and to the right side
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()
