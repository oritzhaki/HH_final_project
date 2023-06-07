
import pandas as pd

def sort(data_path, result_path):
    # read csv into dataframe
    df = pd.read_csv(f"Results/{data_path}")

    # update loss on cell if loss type is l1_loss
    for index, row in df.iterrows():
        if row['CostFunc'] == 'l1_loss':
            df.at[index, 'AvgCost'] = row['AvgCost']**2

    # sort dataframe based on AvgCost column in ascending order
    df = df.sort_values(by='AvgCost')
    
    # save updated dataframe as csv file
    df.to_csv(f'{result_path}/result_sorted_loss.csv', index=False)

    # print message indicating success
    print(f'Updated dataframe saved to {result_path}')
