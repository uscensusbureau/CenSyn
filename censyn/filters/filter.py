import logging
from typing import Dict, List

import pandas as pd


class Filter:
    """
    Base class for filtering of a DataFrame. Filtering produces a DataFrame with a subset of the
    rows based upon some criteria.
    """
    def __init__(self, negate: bool = False) -> None:
        """
        Base initialization for filtering of a DataFrame. This is an identity filter as it will produce
        the exact same as the input DataFrame when negate is False.

        :param: negate: The result DataFrame is to negate of the filter.
        """
        logging.debug(msg=f"Instantiate Filter with negate {negate}.")
        self._negate = negate

    @property
    def negate(self) -> bool:
        """
        Getter for boolean negate.
        """
        return self._negate

    @property
    def dependency(self) -> List[str]:
        """
        The feature dependencies for the filter.

        :return: List of dependent feature names.
        """
        return []

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame. This returns the same DataFrame as the input DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"Filter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        if self.negate:
            mod_df = df.iloc[[]]
        else:
            mod_df = pd.DataFrame(df)
        return mod_df

    def to_dict(self) -> Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        return {'negate': self._negate}
