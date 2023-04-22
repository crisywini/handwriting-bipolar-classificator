import pandas as pd
from sklearn.utils import resample
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    data_unbalanced = pd.read_parquet(
        f'{input_filepath}/bipolar_handwriting_processed.parquet')

    data_balanced = resample_classes(data_unbalanced)
    data_balanced.to_parquet(
        f'{output_filepath}/bipolar_handwriting_processed_balanced.parquet')
    logger.info(f'Data saved {len(data_balanced)}')


def resample_classes(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function allows to resample the classes with less than 100 elements on it
    '''
    count_df = data.groupby(['Label(0,1)'])['Label(0,1)'].count()

    labels_unbalanced = [k for k, v in count_df.items() if v <= 100]

    df_sampled = pd.DataFrame()
    for j in labels_unbalanced:

        df_minority_j = data[data['Label(0,1)'] == j]
        df_minority_upsampled = resample(df_minority_j,
                                         replace=True,
                                         n_samples=400,
                                         stratify=df_minority_j,
                                         random_state=123)
        df_sampled = pd.concat([df_sampled, df_minority_upsampled])
    return pd.concat([data, df_sampled])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    input_url = f'{project_dir}\data\interim'
    output_url = f'{project_dir}\data\processed'
    main(input_url, output_url)
