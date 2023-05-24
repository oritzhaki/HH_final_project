import pandas as pd

def transform_df():
    df = pd.read_csv(f'sample2.csv')
    # Get the number of rows and columns in the original DataFrame
    num_rows, num_cols = df.shape

    # Initialize an empty list to store the new rows
    new_rows = []

    # Iterate over the columns of the original DataFrame
    for col_idx in range(num_cols):
        # Get the column name and corresponding values
        col_name = df.columns[col_idx]
        values = df[col_name].values

        # Iterate over the rows and add the row number, column name, and value to the new rows list
        for row_idx, value in enumerate(values):
            new_row = [row_idx, col_name, value]
            new_rows.append(new_row)

    # Create a new DataFrame from the new rows list
    reshaped_df = pd.DataFrame(new_rows, columns=['0', '1', '2'])

    # Display the result DataFrame
    print(reshaped_df)

    # Save the result DataFrame to a CSV file
    reshaped_df.to_csv(f'sample_2_transformed.csv', index=False)


transform_df()