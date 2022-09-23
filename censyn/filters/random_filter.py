import logging
from typing import Dict

import pandas as pd

from censyn.filters import filter as ft


class RandomFilter(ft.Filter):
    """A Filter that returns a DataFrame based on a random selection."""

    def __init__(self, *args, count: int = 1, proportion: float = None, seed: int = None, **kwargs) -> None:
        """
        A Filter that returns a DataFrame based on a randomization. If proportion is set then the
        filter will return that proportion of records in the DataFrame. Else the filter will return
        the number count of records.

        :param: args: Positional arguments passed on to super.
        :param: count: The number count of records.
        :param: proportion: The proportion of records is a float between 0.0 and 1.0.
        :param: seed: The seed for the random number generator.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate RandomFilter with count {count}, proportion {proportion} and seed {seed}.")
        self._count = count
        self._proportion = proportion
        self._seed = seed

    @property
    def count(self) -> int:
        """The number count of records to filter."""
        return self._count

    @property
    def proportion(self) -> float:
        """The proportion of records to filter."""
        return self._proportion

    @property
    def seed(self) -> int:
        """The seed for the random number generator for the filter."""
        return self._seed

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        :raise: ValueError when the percent is less than 0.0 or larger than 1.0 or when the count is less than 0 or
                larger than the population of the DataFrame.
        """
        logging.debug(msg=f"RandomFilter.execute(). df {df.shape}.")
        # Empty DataFrame
        if df.empty:
            return df

        # Test if performed by percent or number count.
        if self.proportion is not None:
            # Determine percent
            fraction = self.proportion if not self.negate else 1.0 - self.proportion
            return df.sample(frac=fraction, random_state=self.seed)

        # Determine count
        count = self.count if not self.negate else len(df) - self.count
        return df.sample(n=count, random_state=self.seed)

    def to_dict(self) -> Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'count': self._count, 'proportion': self._proportion, 'seed': self._seed}
        attr.update(super().to_dict())
        return attr
