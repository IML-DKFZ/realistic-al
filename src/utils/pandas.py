import pandas as pd


def filter_dataframe(df: pd.DataFrame, filter_dict: dict):
    """
    Filter a pandas DataFrame based on a dictionary with matching names and entries.
    """
    filtered_df = df.copy()  # create a copy of the input DataFrame
    for column, value in filter_dict.items():
        # filter the DataFrame based on the column and value in the dictionary
        filtered_df = filtered_df[filtered_df[column] == value]
    return filtered_df
