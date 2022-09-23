import logging
from typing import Dict, List, Union

import pandas as pd

from .checks import (ChecksActions, CheckDataPrimitive,
                     Postfix_Data_Variable, Postfix_Data_Variable_Name, Postfix_Expression)
from .checks_parser import ChecksParser, ChecksDataFormatExpression


class BaseDataCalculator:
    """BaseDataCalculator class"""

    def __init__(self, expression: Union[str, List[str]], data_format: ChecksDataFormatExpression,
                 feature_name: str = "") -> None:
        """
        BaseDataCalculator initialize

        :param: expression: The data calculation grammar expression string.
        :param: data_format: Data format for the expression to get access function for the grammar.
        :param: feature_name: The feature name to calculate.
        """
        logging.debug(msg=f"Instantiate BaseConsistencyCheck with expression {expression}, "
                          f"data_format {data_format} ans feature_name {feature_name}.")
        self._expr = " ".join(expression) if isinstance(expression, List) else expression
        self._data_format = data_format
        self._feature_name = feature_name

    @property
    def expression(self) -> str:
        """ Expression property. """
        return self._expr

    def compute_variables(self) -> List[str]:
        """
        Compute the variable of the expression.

        :return: List of variable names.
        """
        logging.debug(msg=f"BaseDataCalculator.compute_variables().")
        if not self._data_format:
            msg = "The expression access function for the grammar is None."
            logging.error(msg=msg)
            raise ValueError(msg)

        actions = ChecksActions(df=pd.DataFrame(), has_data=False)
        obj_parser = ChecksParser(data_format_expression=self._data_format, input=self._expr,
                                  actions=actions, types=None)
        obj_parser.parse()

        variables: List[str] = []
        for a in actions.postfix:
            if a.startswith(Postfix_Data_Variable):
                v = a[len(Postfix_Data_Variable):]
                v = v.strip()
                if v not in variables:
                    variables.append(v)
            elif a.startswith(Postfix_Data_Variable_Name):
                v = a[len(Postfix_Data_Variable_Name):]
                v = v.strip()
                if v not in variables:
                    variables.append(v)
            elif a.startswith(Postfix_Expression):
                expr = a[len(Postfix_Expression):]
                expr = expr.strip()
                expr_calculator = AnyDataCalculator(expression=expr,
                                                    data_format=ChecksDataFormatExpression.AnyExpression)
                expr_variable = expr_calculator.compute_variables()
                for var in expr_variable:
                    if var not in variables:
                        variables.append(var)
            elif a.startswith("Data list:"):
                continue
        logging.debug(f"Expression: '{self.expression}' has variables: {variables}")
        return variables

    def execute(self, in_df: pd.DataFrame) -> CheckDataPrimitive:
        """
        Method to perform data generation.

        :param in_df: The input DataFrame data to calculate the data.
        """
        logging.debug(msg=f"BaseDataCalculator.execute(). in_df {in_df.shape}")
        if not self._data_format:
            msg = "The expression access function for the grammar is None."
            logging.error(msg=msg)
            raise ValueError(msg)

        actions = ChecksActions(df=in_df)
        obj_parser = ChecksParser(data_format_expression=self._data_format, input=self._expr,
                                  actions=actions, types=None)
        cur_result = obj_parser.execute()
        if self._feature_name and isinstance(cur_result, pd.Series):
            cur_result.name = self._feature_name
        return cur_result

    def to_dict(self) -> Dict:
        """The DataCalculator's attributes in dictionary for use in processor configuration."""
        return {
            'expression': self._expr,
            'data_format': self._data_format,
            'feature_name': self._feature_name,
        }


class BooleanDataCalculator(BaseDataCalculator):
    """BooleanDataCalculator class"""
    def __init__(self, *args, **kwargs) -> None:
        """
        BooleanDataCalculator initialize

        :param: feature_name: The feature name to calculate.
        :param: expression: The check expression string.
        """
        if 'data_format' not in kwargs.keys():
            kwargs['data_format'] = ChecksDataFormatExpression.BooleanExpression
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate BooleanDataCalculator.")


class NumericDataCalculator(BaseDataCalculator):
    """NumericDataCalculator class"""
    def __init__(self, *args, **kwargs) -> None:
        """
        NumericDataCalculator initialize

        :param feature_name: The feature name to calculate.
        :param expression: The check expression string.
        """
        if 'data_format' not in kwargs.keys():
            kwargs['data_format'] = ChecksDataFormatExpression.NumericExpression
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate NumericDataCalculator.")


class StringDataCalculator(BaseDataCalculator):
    """StringDataCalculator class"""
    def __init__(self, *args, **kwargs) -> None:
        """
        StringDataCalculator initialize

        :param: feature_name: The feature name to calculate.
        :param: expression: The check expression string.
        """
        if 'data_format' not in kwargs.keys():
            kwargs['data_format'] = ChecksDataFormatExpression.StringExpression
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate StringDataCalculator.")


class AnyDataCalculator(BaseDataCalculator):
    """AnyDataCalculator class"""

    def __init__(self, *args, **kwargs) -> None:
        """
        AnyDataCalculator initialize

        :param: feature_name: The feature name to calculate.
        :param: expression: The check expression string.
        """
        if 'data_format' not in kwargs.keys():
            kwargs['data_format'] = ChecksDataFormatExpression.AnyExpression
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate AnyDataCalculator.")
