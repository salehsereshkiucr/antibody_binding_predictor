import pandas as pd

def read_combined():
    return pd.read_csv('./data/Combineddf.tsv', sep='\t')

def get_df_summary(df):
    summary = df.describe(include='all')
    return summary.loc[['min', 'max', 'mean']].T

