import logging

import pandas as pd
import pyreadstat

from .datasource import DataSource, FileSource


class SasDataSource(DataSource):
    """SAS data file loader"""

    @FileSource.validate_extension(extensions='.sas7bdat')
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for the Parquet data source.

        :raise: ValueError: if the file name is not an existing valid file with the valid extension.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate SasDataSource.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Create a Pandas DataFrame from the data source.  Reads the parquet from the file name to generate a DataFrame.

        :return: Created Pandas DataFrame.
        """
        logging.debug(msg=f"SasDataSource.to_dataframe().")
        if self.columns:
            data, meta = pyreadstat.read_sas7bdat(filename_path=self.path_to_file, encoding='iso-8859-1',
                                                  usecols=self.columns)
        else:
            data, meta = pyreadstat.read_sas7bdat(filename_path=self.path_to_file, encoding='iso-8859-1')
        # Set empty strings to None
        for col in data.columns:
            if data[col].dtype == 'object':
                mask = (data[col] == "")
                if any(mask):
                    col_data = data[col].copy()
                    col_data.loc[mask] = None
                    data[col] = col_data
        return data

    @staticmethod
    def save_dataframe(file_name: str, in_df: pd.DataFrame) -> None:
        """
        Saves a data frame to a data source.

        :param: file_name: File name to save the data set.
        :param: in_df: Pandas DataFrame to save
        :return: None
        """
        logging.debug(msg=f"SasDataSource.save_dataframe().")
        raise NotImplementedError('Attempted call to unsupported method SasDataSource.save_dataframe()')
