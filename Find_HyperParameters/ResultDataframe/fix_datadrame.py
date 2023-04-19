import pandas as pd

# read csv into dataframe
df = pd.read_csv('my_dataframe.csv')

# calculate average of AllCost column
df['AvgCost'] = df['AllCost'].apply(lambda x: sum(map(float, x.strip('[]').split(','))) / len(x.strip('[]').split(',')))

# save updated dataframe as csv file
df.to_csv('my_dataframe_updated.csv', index=False)

# print message indicating success
print('Updated dataframe saved to my_dataframe_updated.csv')
