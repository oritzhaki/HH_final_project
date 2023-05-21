import pandas as pd

def transform_df(i,num):
    df = pd.read_csv(f'DataCleaning/dataset/sample_{num}.csv')
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
    reshaped_df.to_csv(f'Production/ProdDataRats/sample_{i}.csv', index=False)


wanted_files = [9,14,15,20,23,48,30,52,45,68]
i = 1
for f in wanted_files:
    transform_df(i, f)
    i+=1