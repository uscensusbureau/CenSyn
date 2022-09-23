import logging
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd

from .checks_peg import FAILURE, format_error, Parser, ParseError, TreeNode

# Types of objects expected for a data value
CheckDataPrimitive = Union[None, bool, int, float, str, List, pd.Series]


class ChecksDataFormatExpression(Enum):
    """This is an enumeration that represents the different data types of expressions."""
    ChecksExpression = 0
    BooleanExpression = 1
    NumericExpression = 2
    StringExpression = 3
    AnyExpression = 4


class ChecksParser(Parser):
    def __init__(self, *args, data_format_expression: ChecksDataFormatExpression, **kwargs) -> None:
        """
        Check parser.

        :param: args:  Standard arguments.
        :param: data_format_expression: Data format for the expression to get access function for the grammar.
        :param: kwargs:  Standard keyword arguments.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate ChecksParser with data_format_expression {data_format_expression}.")

        self._actions = kwargs.get('actions', None)
        self._data_format_expression = data_format_expression

        dataformat_dict = {
            ChecksDataFormatExpression.ChecksExpression: "_read_ChecksExpression",
            ChecksDataFormatExpression.BooleanExpression: "_read_BooleanExpression",
            ChecksDataFormatExpression.NumericExpression: "_read_NumericExpression",
            ChecksDataFormatExpression.StringExpression: "_read_StringExpression",
            ChecksDataFormatExpression.AnyExpression: "_read_AnyExpression",
        }
        access_function = dataformat_dict[data_format_expression]
        self._access_func = getattr(self, access_function)

    def parse(self) -> TreeNode:
        """ Parse function called from base Parser. """
        logging.debug(msg=f"ChecksParser.parse().")
        tree = self._access_func()
        if tree is not FAILURE and self._offset == self._input_size:
            return tree
        if not self._expected:
            self._failure = self._offset
            self._expected.append('<EOF>')
        raise ParseError(format_error(self._input, self._failure, self._expected))

    def execute(self) -> CheckDataPrimitive:
        """
        Method to perform data generation.
        """
        logging.debug(msg=f"ChecksParser.execute().")
        self.parse()
        if len(self._actions.data) != 1:
            raise ValueError(f"invalid stack size {len(self._actions.data)} for result.")
        cur_result = self._actions.data.pop()

        if not isinstance(cur_result, pd.Series):
            if self._data_format_expression == ChecksDataFormatExpression.ChecksExpression:
                if not isinstance(cur_result, bool):
                    msg = f"Invalid data type {type(cur_result)} for result."
                    logging.error(msg=msg)
                    raise ValueError(msg)
            elif self._data_format_expression == ChecksDataFormatExpression.BooleanExpression:
                if not isinstance(cur_result, bool):
                    msg = f"Invalid data type {type(cur_result)} for result."
                    logging.error(msg=msg)
                    raise ValueError(msg)
            elif self._data_format_expression == ChecksDataFormatExpression.NumericExpression:
                if not isinstance(cur_result, (int, float, complex, np.number)):
                    msg = f"Invalid data type {type(cur_result)} for result."
                    logging.error(msg=msg)
                    raise ValueError(msg)
                if not isinstance(cur_result, (int, float, complex)):
                    cur_result = int(cur_result)
            elif self._data_format_expression == ChecksDataFormatExpression.StringExpression:
                if not isinstance(cur_result, str):
                    msg = f"Invalid data type {type(cur_result)} for result."
                    logging.error(msg=msg)
                    raise ValueError(msg)
            if self._actions.input_df is not None and not self._actions.input_df.empty:
                cur_result = pd.Series(index=self._actions.input_df.index, data=cur_result)
        return cur_result
