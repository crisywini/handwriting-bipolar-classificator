import pandas as pd
import numpy as np


def process(input_filepath: str, output_filepath: str) -> pd.DataFrame:
    data = get_dataset(input_filepath)
    data_processed = process_data(data)
    data_processed.to_parquet(
        f'{output_filepath}/bipolar_handwriting_processed.parquet', index=False)

    return data_processed


def get_dataset(input_filepath: str) -> pd.DataFrame:
    return pd.read_csv(f'{input_filepath}/Original_Dataset.csv')


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return (data
            .pipe(drop_exact_duplicates)
            .pipe(clean_numeric_columns)
            .pipe(set_numeric_column_types)
            .pipe(set_categoric_column_types))


def drop_exact_duplicates(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to delete all the exact duplicated rows in the dataset
    '''
    return data_frame.drop_duplicates()


def clean_numeric_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to clean the numeric columns
    '''

    data_frame.loc[((data_frame['V(Sx)'] == '39/ 55')), ['V(Sx)']] = np.nan
    return data_frame.dropna()


def set_numeric_column_types(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to set the numeric columns types of the dataset
    '''
    data_frame['V(Sx)'] = data_frame['V(Sx)'].astype('float')
    data_frame['V(L)'] = data_frame['V(L)'].astype('float')
    return data_frame


def set_categoric_column_types(data_frame: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to set the categoric columns types of the dataset
    '''
    data_frame['Men'] = data_frame['Men'].astype('category')
    data_frame['Femal'] = data_frame['Femal'].astype('category')
    data_frame['Age(0,0.5,1)'] = data_frame['Age(0,0.5,1)'].astype('category')
    data_frame['Label(0,1)'] = data_frame['Label(0,1)'].astype('category')
    return data_frame
