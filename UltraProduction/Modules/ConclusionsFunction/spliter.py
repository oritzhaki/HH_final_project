
import pandas as pd

def split_good_bad_runs(result_path, save_size, return_size):

    SIZE_TO_SAVE = save_size
    SIZE_TO_RETURN = return_size

    # Load the original CSV
    df = pd.read_csv(f'{result_path}/result_sorted_loss.csv')

    # Create two dataframes based on the condition on AvgCost
    good_runs = df.head(SIZE_TO_SAVE)
    bad_runs = df.tail(len(df)-SIZE_TO_SAVE)  # or simply: bad_runs = df[10:]

    # Save these dataframes to new CSV files
    good_runs.to_csv(f'{result_path}/GoodRuns.csv', index=False)
    bad_runs.to_csv(f'{result_path}/BadRuns.csv', index=False)
    
    good_runs = df.head(SIZE_TO_RETURN)
    bad_runs = df.tail(len(df)-SIZE_TO_RETURN)  # or simply: bad_runs = df[10:]
    
    return good_runs, bad_runs