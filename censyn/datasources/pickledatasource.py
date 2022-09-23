import logging

import pandas as pd

from .datasource import DataSource, FileSource


class PickleDataSource(DataSource):
    """
    Pickle Data file loader. This will read a pickle file and create a pandas dataframe.
    """

    @FileSource.validate_extension(extensions='.pickle')
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for the Pickle data source.
        
        :param columns: List of columns names to be read from the file. When columns is None then all columns will be
        read. Default=None.
        :raise ValueError: if the file name is not an existing valid file with a '.parquet' extension.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate PickleDataSource.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Creates a Pandas Dataframe using the full_file_name provided to the constructor.
        This will also be filtered based on if you provided a list of columns in the constructor.

        :return: Created Pandas DataFrame.
        """
        logging.debug(msg=f"PickleDataSource.to_dataframe().")
        data_frame = pd.read_pickle(self.path_to_file)
        data_frame = data_frame[self.columns] if self.columns else data_frame
        return data_frame

    @staticmethod
    def save_dataframe(file_name: str, in_df: pd.DataFrame) -> None:
        """
        Saves a data frame to a data source.

        :param file_name: File name to save the data set.
        :param in_df: Pandas DataFrame to save
        :return: None
        """
        logging.debug(msg=f"PickleDataSource.save_dataframe().")
        in_df.to_pickle(path=file_name)
