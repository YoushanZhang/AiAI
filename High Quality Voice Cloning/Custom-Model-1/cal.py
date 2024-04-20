import glob
import pandas as pd
# # Use glob to search for files starting with the specified prefix
# matching_files = glob.glob('.' + "/Val_Metric*.csv")

# dfs=[]
# for file in matching_files:
#     df = pd.read_csv(file)
#     dfs.append(df)
#     # print(dfs)

# concatenated_df = pd.concat(dfs, ignore_index=True)


# concatenated_df.to_csv('valid_metric.csv',index=False)

df = pd.read_csv('valid_metric.csv')
print(df)
print(df.describe())