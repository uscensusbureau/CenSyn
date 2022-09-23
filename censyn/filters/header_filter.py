import logging
from typing import Dict, List

import pandas as pd

from .filter import Filter


class HeaderFilter(Filter):
    """A Filter that takes a DataFrame, List of feature names and returns a filtered DataFrame."""
    def __init__(self,  headers: List[str], *args, **kwargs) -> None:
        """
        Header initialization for filtering of DataFrames based on the comparison of header values.

        :param: headers: These are the feature names that will be filtered from your dataframe.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate HeaderFilter with headers {headers}.")
        self._headers = headers
        if headers is None:
            msg = f"HeaderFilter requires a list of features you passed none."
            logging.error(msg=msg)
            raise ValueError(msg)

    def execute(self, df: pd.DataFrame):
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"HeaderFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.negate:
            return df.drop(columns=self._headers)
        return df[self._headers]

    @property
    def headers(self) -> List[str]:
        return self._headers

    @property
    def dependency(self) -> List[str]:
        """
        The feature dependencies for the filter.

        :return: List of dependent feature names.
        """
        return self._headers

    def to_dict(self) -> Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'headers': self._headers}
        attr.update(super().to_dict())
        return attr
