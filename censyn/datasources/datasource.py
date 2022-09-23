import logging
from abc import abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

from .filesource import FileSource


class DataSource(FileSource):
    """Abstract class for handling the loading of panda DataFrames from input files"""

    def __init__(self, columns: Union[List[str], None] = None, row_count: int = None, *args, **kwargs) -> None:
        """
        Initialization for DataSource.

        :param: columns: List of columns names to be read from the file. When columns is None then all columns will be
        read. Default=None.
        :param: row_count: The number of rows that the datasource will read. If None is provided it will read the entire
            file.
        :param: args: Standard arguments.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate DataSource with columns {columns} and row_count {row_count}.")
        self._columns = columns
        self._row_count = row_count

    @property
    def columns(self) -> Union[List[str], None]:
        """ Columns property. """
        return self._columns

    @property
    def row_count(self) -> int:
        """ Row count property. """
        return self._row_count

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Abstract method for returning a data frame from a data source."""
        raise NotImplementedError('Attempted call to abstract method DataSource.to_dataframe()')

    @staticmethod
    @abstractmethod
    def save_dataframe(file_name: str, in_df: pd.DataFrame) -> None:
        """
        Abstract method for saving a data frame to a data source.

        :param: file_name: File name to save the data set.
        :param: in_df: Pandas DataFrame to save
        :return: None
        """
        raise NotImplementedError('Attempted call to abstract method DataSource.save_dataframe()')


class DelimitedDataSource(DataSource):
    """
    Handles the loading of data into a pandas Dataframe from any csv-like file, for example csv, tsv, or psv.
    By default, it assumes comma-separated with headers on row 0.
    """

    def __init__(self, separator: str = ',', comment_line: str = '#', header: Union[List, int] = 0, *args, **kwargs):
        """
        Initializes the DelimitedDataSource.

        :param: separator: The data separator, eg ',' or '|'. default ','
        :param: comment_line: A character to demarcate comments from data. default '#'
        :param: header: either an int corresponding to the header line number, or a list of headers. default 0.
        :param: args: Standard arguments.
        :param: kwargs: Standard keyword arguments.
        :raise: TypeError: if header is not an int or list
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate DataSource with separator {separator}, comment_line {comment_line} and "
                          f"header {header}.")

        if not isinstance(header, (int, list)):
            raise TypeError(f'Header must be of type int or list, got {type(header)}')

        self._separator = separator
        self._comment_line = comment_line
        self._header = header

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a dataframe from the given file."""
        logging.debug(msg=f"DelimitedDataSource.to_dataframe().")
        if isinstance(self._header, int):
            data_frame = pd.read_csv(self.file_handle, sep=self._separator, comment=self._comment_line,
                                     nrows=self.row_count, header=self._header)
        else:
            data_frame = pd.read_csv(self.file_handle, sep=self._separator, comment=self._comment_line,
                                     nrows=self.row_count, names=self._header)
        data_frame = data_frame[self.columns] if self.columns else data_frame

        for f_name in data_frame.columns:
            if np.issubdtype(data_frame[f_name].dtype, np.number):
                pass
            elif data_frame[f_name].dtype == 'object':
                mod_s = pd.Series(data=data_frame[f_name].apply(lambda x: str(x) if x == x else None),
                                  dtype=object)
                data_frame[f_name] = mod_s
            else:
                cur_data_type = data_frame[f_name].dtype
                raise ValueError(f'Unsupported data type {cur_data_type} for feature {data_frame[f_name].name}.')

        return data_frame

    @staticmethod
    def save_dataframe(file_name: str, in_df: pd.DataFrame) -> None:
        """
        Saves a data frame to a data source.

        :param: file_name: File name to save the data set.
        :param: in_df: Pandas DataFrame to save
        :return: None
        """
        logging.debug(msg=f"DelimitedDataSource.save_dataframe().")
        in_df.to_csv(path_or_buf=file_name)
