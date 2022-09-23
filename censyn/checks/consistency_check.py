import logging
from typing import List

import pandas as pd

from .checks_data_calculator import BooleanDataCalculator
from .checks_parser import ChecksDataFormatExpression


class ConsistencyCheck(BooleanDataCalculator):
    """Consistency check class"""
    def __init__(self, *args, **kwargs) -> None:
        """
        Consistency Check initialize

        :param: expression: The check expression string.
        """
        if 'data_format' not in kwargs.keys():
            kwargs['data_format'] = ChecksDataFormatExpression.ChecksExpression
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate ConsistencyCheck.")

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to perform consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: The List of the indexes which fail the consistency check.
        """
        logging.debug(msg=f"ConsistencyCheck.execute(). in_df {in_df.shape}.")
        cur_result = super().execute(in_df=in_df)
        return in_df.index[~cur_result]
