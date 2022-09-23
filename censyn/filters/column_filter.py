import logging
import math
from abc import abstractmethod
from typing import Any, Dict, List

import pandas as pd

from censyn.filters import filter as ft


class ColumnFilter(ft.Filter):
    """A Filter that returns a DataFrame based on a column value."""
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Base column initialization for filtering of DataFrames based on the comparison of column values.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for filtering.
        :param: value: The value for testing of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate ColumnFilter with header {header}, value {value}.")
        self._header = header
        self._value = value

    @property
    def header(self) -> str:
        """The column header for performing filtering upon."""
        return self._header

    @property
    def value(self) -> Any:
        """The value for comparing of the table's column values."""
        return self._value

    @property
    def dependency(self) -> List[str]:
        """
        The feature dependencies for the filter.

        :return: List of dependent feature names.
        """
        return [self._header]

    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'header': self._header, 'value': self._value}
        attr.update(super().to_dict())
        return attr


class ColumnIsnullFilter(ColumnFilter):
    """A Filter that returns a DataFrame based on a column's entries are null."""
    def __init__(self, *args, header: str, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        null.  Produces a subset of the input DataFrame with just the rows having
        the column header's entries are null.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=None, **kwargs)
        logging.debug(msg=f"Instantiate ColumnIsnullFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnIsnullFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.negate:
            return df[~df[self.header].isnull()]
        return df[df[self.header].isnull()]


class ColumnEqualsFilter(ColumnFilter):
    """A Filter that returns a DataFrame based on a column's entries equal to the value."""
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        equal to the value.  Produces a subset of the input DataFrame with just the rows having
        the column header's entries equal to the value.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The value for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnEqualsFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnEqualsFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.value is None:
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if not isinstance(self.value, str) and math.isnan(self.value):
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if self.negate:
            return df[df[self.header] != self.value]
        return df[df[self.header] == self.value]


class ColumnEqualsListFilter(ColumnFilter):
    """A Filter that returns a DataFrame based on a column's entries equal to an item in the value list."""
    def __init__(self, *args, header: str, value: List, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        equal to an item in the value list.  Produces a subset of the input DataFrame with just the rows
        having the column header's entries contained in the value list.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The list of values for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnEqualsListFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnEqualsListFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        mask = pd.Series(data=False, index=df.index)
        for v in self.value:
            if v is None:
                mask = mask | df[self.header].isna()
            elif not isinstance(v, str) and math.isnan(v):
                mask = mask | df[self.header].isna()
            else:
                mask = mask | (df[self.header] == v)
        if self.negate:
            mask = mask.apply(lambda x: x is not True)
        return df[mask]


class ColumnGreaterThanFilter(ColumnFilter):
    """
    A Filter that returns a DataFrame based on a column is greater than the value.
    """
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        greater than the value.  Produces a subset of the input DataFrame with just the rows having
        the column header's entries greater than the value.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The value for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnGreaterThanFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnGreaterThanFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.value is None:
            if self.negate:
                return df
            return pd.DataFrame(columns=df.columns)
        if not isinstance(self.value, str) and math.isnan(self.value):
            if self.negate:
                return df
            return pd.DataFrame(columns=df.columns)
        if self.negate:
            return df[~(df[self.header] > self.value)]
        return df[df[self.header] > self.value]


class ColumnGreaterThanEqualFilter(ColumnFilter):
    """
    A Filter that returns a DataFrame based on a column is greater than or equal to the value.
    """
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        greater than or equal to the value.  Produces a subset of the input DataFrame with just
        the rows having the column header's entries greater than or equal to the value.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The value for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnGreaterThanEqualFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnGreaterThanEqualFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.value is None:
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if not isinstance(self.value, str) and math.isnan(self.value):
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if self.negate:
            return df[~(df[self.header] >= self.value)]
        return df[df[self.header] >= self.value]


class ColumnLessThanFilter(ColumnFilter):
    """
    A Filter that returns a DataFrame based on a column is less than the value.
    """
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        less than the value.  Produces a subset of the input DataFrame with just the rows having
        the column header's entries less than the value.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The value for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnLessThanFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnLessThanFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.value is None:
            if self.negate:
                return df
            return pd.DataFrame(columns=df.columns)
        if not isinstance(self.value, str) and math.isnan(self.value):
            if self.negate:
                return df
            return pd.DataFrame(columns=df.columns)
        if self.negate:
            return df[~(df[self.header] < self.value)]
        return df[df[self.header] < self.value]


class ColumnLessThanEqualFilter(ColumnFilter):
    """
    A Filter that returns a DataFrame based on a column is less or equal than the value.
    """
    def __init__(self, *args, header: str, value: Any, **kwargs) -> None:
        """
        Initialization for filtering of DataFrames based on a column as identified by header is
        less than or equal to the value. Produces a subset of the input DataFrame with just the
        rows having the column header's entries less than or equal to the value.

        :param: args: Positional arguments passed on to super.
        :param: header: The column header for performing filtering upon.
        :param: value: The value for comparison of the column.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, header=header, value=value, **kwargs)
        logging.debug(msg=f"Instantiate ColumnLessThanEqualFilter.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ColumnLessThanEqualFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.value is None:
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if not isinstance(self.value, str) and math.isnan(self.value):
            if self.negate:
                return df[~df[self.header].isna()]
            return df[df[self.header].isna()]
        if self.negate:
            return df[~(df[self.header] <= self.value)]
        return df[df[self.header] <= self.value]
