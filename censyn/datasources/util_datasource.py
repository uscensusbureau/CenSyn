import logging
import pathlib
import typing

import pandas as pd

from .datasource import DelimitedDataSource
from .parquetdatasource import ParquetDataSource
from .pickledatasource import PickleDataSource
from .sasdatasource import SasDataSource

data_sources_dict = {
    'csv': DelimitedDataSource,
    'parquet': ParquetDataSource,
    'pickle': PickleDataSource,
    'sas7bdat': SasDataSource
}


def load_data_source(file_names: typing.List[str], feature_names: typing.List[str] = None) -> pd.DataFrame:
    """
    Create a Pandas DataFrame from the data source.  Reads the parquet from the file name to generate a DataFrame.
    Supported sources include: Parquet, CSV, Pickle and SAS.

    :param: file_names: The list of file names for the data source.
    :param: feature_names: List of columns or feature names to be read from the file. When columns is None then all
        columns will be read. Default=None.
    :raise: ValueError: if the file name is not an existing valid file with a supported extension.
    :return: Created Pandas DataFrame.
    """
    logging.debug(msg=f"util_datasource.load_data_source(). file_names {file_names} and feature_names {feature_names}.")
    to_return = []

    for file_name in file_names:
        logging.info(f'Loading data file {file_name}')
        parts = file_name.split('.')
        ds = data_sources_dict.get(parts[-1])
        if not ds:
            msg = f"Must provide a supported data source. Provide either a parquet, csv, pickle or " \
                  f"SAS' + (.sas7bdat) file. Found {file_name} instead."
            logging.error(msg=msg)
            raise ValueError(msg)
        cur_df = ds(path_to_file=file_name).to_dataframe()
        if feature_names:
            # Set order of columns the order of the features.
            columns = [c for c in feature_names if c in cur_df.columns]
            cur_df = cur_df[columns]
        to_return.append(cur_df)

    return pd.concat(to_return, axis=0, join='outer', ignore_index=False).reset_index(drop=True)


def save_data_source(file_name: str, in_df: pd.DataFrame) -> None:
    """
    Saves a data frame to a data source.

    :param: file_name: File name to save the data set.
    :param: in_df: Pandas DataFrame to save
    :return: None
    """
    logging.debug(msg=f"util_datasource.load_data_source(). file_name {file_name} and in_df {in_df.shape}.")
    logging.info(f'Saving data file {file_name}')
    parts = file_name.split('.')
    ds = data_sources_dict.get(parts[-1])
    if not ds:
        msg = f"Must provide a supported data source. Provide either a parquet, csv, pickle or " \
              f"SAS' + (.sas7bdat) file. Found {file_name} instead."
        logging.error(msg=msg)
        raise ValueError(msg)
    ds.save_dataframe(file_name, in_df)


def validate_data_source(file_name: str) -> bool:
    """
    Validate a data source. It checks that the file name is a supported data source and if the path exists.

    :param: file_name: File name to validate the data set.
    :return: Boolean flag if it is a supported data source.
    """
    logging.debug(msg=f"util_datasource.validate_data_source(). file_name {file_name}.")
    parts = file_name.split('.')
    ds = data_sources_dict.get(parts[-1])
    if not ds:
        return False
    path = pathlib.Path(file_name).parent
    if path and not path.exists():
        return False
    return True
