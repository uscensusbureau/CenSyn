import logging
from typing import Dict, List, Union

import pandas as pd

from censyn.checks.checks_data_calculator import BooleanDataCalculator
from .filter import Filter


class ExpressionFilter(Filter):
    """A Filter that returns a DataFrame based on a column value."""
    def __init__(self, *args, expr: Union[str, List[str]], **kwargs) -> None:
        """
        Base column initialization for filtering of DataFrames based on the Data Calculation grammar expression.

        :param: args: Positional arguments passed on to super.
        :param: expr: Data Calculation grammar expression.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate ExpressionFilter with expr {expr}.")
        if isinstance(expr, List):
            expr = " ".join(expr)
        self._expr = expr

    @property
    def expression(self) -> str:
        """The expression for performing filtering upon."""
        return self._expr

    @property
    def dependency(self) -> List[str]:
        """
        The feature dependencies for the filter.

        :return: List of dependent feature names.
        """
        logging.debug(msg=f"ExpressionFilter.dependency().")
        calculator = BooleanDataCalculator(feature_name="", expression=self._expr)
        return calculator.compute_variables()

    def execute(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform filter execution on DataFrame.

        :param: in_df: The DataFrame the filter is performed upon.
        :return: The resulting filtered DataFrame.
        """
        logging.debug(msg=f"ExpressionFilter.execute(). in_df {in_df.shape}.")
        # Empty DataFrame
        if in_df.empty:
            return in_df

        calculator = BooleanDataCalculator(feature_name="", expression=self._expr)
        cur_result = calculator.execute(in_df=in_df)
        if not isinstance(cur_result, pd.Series):
            if not isinstance(cur_result, bool):
                raise ValueError(f"Invalid data type {type(cur_result)} for result.")
            cur_result = pd.Series(index=in_df.index, data=cur_result)
        if self.negate:
            return in_df[cur_result]
        return in_df[cur_result]

    def to_dict(self) -> Dict:
        """The Filter's attributes in dictionary for use in processor configuration."""
        attr = {'expression': self._expr}
        attr.update(super().to_dict())
        return attr
