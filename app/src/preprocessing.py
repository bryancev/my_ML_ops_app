# Import libraries
import pandas as pd
import numpy as np

# Define column types

drop_col = ['client_id', 'mrg_', 'регион', 'использование',
            'частота', 'зона_1', 'зона_2', 'pack',  'сегмент_arpu']

target_col = 'binary_target'

def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=drop_col)
    return input_df

# Main preprocessing function
def run_preproc(input_df):
    # Create dataframe and feature engeeneer
    input_df["сумма"].fillna(-1000, inplace=True)
    input_df["продукт_1"].fillna(-1, inplace=True)
    input_df["продукт_2"].fillna(-1, inplace=True)

    input_df["count_of_NaNs"] = input_df.isna().sum(axis=1)
    input_df['пропуск_суммы'] = np.where(input_df['сумма'].isna(), 1, 0)
    input_df['эффективность_использования'] = input_df['доход'] / (input_df['сумма'] + 1e-4)

    output_df = input_df

    # Return resulting dataset
    return output_df