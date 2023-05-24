import pandas as pd
import matplotlib.pyplot as plt

# Define the colors you want to use
colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "lime", "pink"]

# Read the two dataframes
based_generated = pd.read_csv('GeneratedDataBasedGAParams/dataset_based_ga_params.csv', header=None)
based_generated = based_generated.drop(based_generated.index[0])
Test_df = pd.read_csv('DatasetTest/test.csv', header=None)
Test_df = Test_df.drop(Test_df.index[0])

based_generated.columns = ['Time', 'Temp', 'Value']
Test_df.columns = ['Time', 'Temp', 'Value']

# Convert 'Time', 'Temp' and 'Value' to numeric types
based_generated['Time'] = pd.to_numeric(based_generated['Time'])
based_generated['Temp'] = pd.to_numeric(based_generated['Temp'])
based_generated['Value'] = pd.to_numeric(based_generated['Value'])
Test_df['Time'] = pd.to_numeric(Test_df['Time'])
Test_df['Temp'] = pd.to_numeric(Test_df['Temp'])
Test_df['Value'] = pd.to_numeric(Test_df['Value'])

# Create a dictionary mapping Temp values to colors (wrap around the color list if more Temp values than colors)
temp_values = sorted(based_generated["Temp"].unique())
temp_color = {temp: colors[i % len(colors)] for i, temp in enumerate(temp_values)}

# Plot df1 with smaller markers
for temp in based_generated['Temp'].unique():
    plt.plot(based_generated.loc[based_generated['Temp'] == temp, 'Time'], 
             based_generated.loc[based_generated['Temp'] == temp, 'Value'], 'x', 
             label=f'Based On GA Params, Temp={temp}', 
             color=temp_color[temp], markersize=1)

# Plot df2 with smaller markers
for temp in Test_df['Temp'].unique():
    plt.plot(Test_df.loc[Test_df['Temp'] == temp, 'Time'], 
             Test_df.loc[Test_df['Temp'] == temp, 'Value'], 'o', 
             label=f'Test (Truth + Noise), Temp={temp}', 
             color=temp_color[temp], markersize=1)

# Add title and labels
plt.title('Dataset Comparison')
plt.xlabel('Time (ms)')
plt.ylabel('Conductivity (G)')


# Show the plot
plt.show()
