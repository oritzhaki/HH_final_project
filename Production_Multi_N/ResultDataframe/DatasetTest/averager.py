import pandas as pd

# Create the two DataFrames
df1 = pd.read_csv('sample_9_removed.csv')
df2 = pd.read_csv('sample_10_removed.csv')

# Calculate the average between the two DataFrames
avg_df = (df1 + df2) / 2

# Print the average DataFrame
avg_df.to_csv("test.csv", index=False)