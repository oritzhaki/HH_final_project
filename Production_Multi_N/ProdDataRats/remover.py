import pandas as pd
import os

# Create the output directory if it doesn't exist
output_dir = 'ProdDataRatsRemoved'
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(f'sample_2_transformed.csv')

# Filter rows where the value in the second column is less than -20
df = df[df.iloc[:, 1] >= -20]

# Filter rows where the value in the first column is between 3 and 53
df = df[(df.iloc[:, 0] >= 0) & (df.iloc[:, 0] <= 49)]

# Save the filtered DataFrame to a new file in the output directory
df.to_csv(f'{output_dir}/sample_9_removed_1.csv', index=False)
