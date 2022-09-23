import logging

import pandas as pd

from .datasource import DataSource, FileSource


class ParquetDataSource(DataSource):
    """Parquet data file loader"""

    @FileSource.validate_extension(extensions='.parquet')
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for the Parquet data source.

        :raise: ValueError: if the file name is not an existing valid file with a '.parquet' extension.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate ParquetDataSource.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a Pandas DataFrame from the data source.  Reads the parquet from the file name to generate a DataFrame.

        :return: Created Pandas DataFrame.
        """
        logging.debug(msg=f"ParquetDataSource.to_dataframe().")
        return pd.read_parquet(path=self.path_to_file, columns=self.columns)

    @staticmethod
    def save_dataframe(file_name: str, in_df: pd.DataFrame) -> None:
        """
        Saves a data frame to a data source.

        :param file_name: File name to save the data set.
        :param in_df: Pandas DataFrame to save
        :return: None
        """
        logging.debug(msg=f"ParquetDataSource.save_dataframe(). file_name {file_name} and in_df {in_df.shape}.")
        in_df.to_parquet(path=file_name)
