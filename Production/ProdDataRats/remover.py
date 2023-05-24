import pandas as pd
import os

# Create the output directory if it doesn't exist
output_dir = 'ProdDataRatsRemoved'
os.makedirs(output_dir, exist_ok=True)

# Loop over the file indices
for i in range(1, 11):
    # Read the current file
    df = pd.read_csv(f'sample_{i}.csv')

    # Filter rows where the value in the second column is less than -20
    df = df[df.iloc[:, 1] >= -20]

    # Filter rows where the value in the first column is between 3 and 53
    df = df[(df.iloc[:, 0] >= 0) & (df.iloc[:, 0] <= 49)]

    # Save the filtered DataFrame to a new file in the output directory
    df.to_csv(f'{output_dir}/sample_{i}_removed.csv', index=False)
