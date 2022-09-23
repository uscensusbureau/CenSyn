import logging
import typing

import pandas as pd

from censyn.filters import filter as ft


class IndexFilter(ft.Filter):
    """A Filter that returns a DataFrame based on the integer location based indexes."""
    def __init__(self, *args, indexes: typing.List[int], **kwargs) -> None:
        """
        A Filter that returns a DataFrame based on a list of indexes.

        :param: args: Positional arguments passed on to super.
        :param: indexes: List of indexes.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate IndexFilter with indexes {indexes}.")
        if indexes is None:
            msg = f"Indexes must not be None."
            logging.error(msg=msg)
            raise ValueError(msg)
        self._indexes = indexes

    @property
    def indexes(self) -> typing.List[int]:
        """The list of indexes of records to filter."""
        return self._indexes

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        :raise: IndexError when an index is out of bounds of the DataFrame.
        """
        logging.debug(msg=f"IndexFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.negate:
            mod_df = df.loc[~df.index.isin(self._indexes)]
        else:
            mod_df = df.iloc[self._indexes]
        return mod_df

    def to_dict(self) -> typing.Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'indexes': self._indexes}
        attr.update(super().to_dict())
        return attr


class IndexRangeFilter(ft.Filter):
    """A Filter that returns a range of rows back based on the integer location based indexes."""
    def __init__(self, *args, start: int = 0, end: int = 1, **kwargs) -> None:
        """
        A Filter that returns a DataFrame based on a range of indexes.

        :param: args: Positional arguments passed on to super.
        :param: start: Start index.
        :param: end: End index.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate IndexRangeFilter with start {start} and end {end}.")
        self._start = start
        self._end = end

    @property
    def start(self) -> int:
        """The start index of records to filter."""
        return self._start

    @property
    def end(self) -> int:
        """The end index of records to filter."""
        return self._end

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"IndexRangeFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.negate:
            b = df.index.isin(df.index[self._start: self._end])
            mod_df = df.loc[~b]
        else:
            mod_df = df.iloc[self._start: self._end]
        return mod_df

    def to_dict(self) -> typing.Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'start': self._start, 'end': self._end}
        attr.update(super().to_dict())
        return attr
