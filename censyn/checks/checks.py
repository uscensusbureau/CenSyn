import inspect
import logging
from typing import List, Union

import numpy as np
import pandas as pd
import pandas.core.groupby as pdgroup

from .checks_peg import FAILURE, ParseError, TreeNode
from .checks_parser import ChecksParser, ChecksDataFormatExpression, CheckDataPrimitive


Postfix_Expression = "Expression:"
Postfix_Data_Variable = "Data variable:"
Postfix_Data_Variable_Name = "Data variable name:"
keywords = ["or", "xor", "and", "not", "equal", "notequal", "lessequal", "less",
            "greaterequal", "greater", "equalstr", "notequalstr", "is",
            "abs", "min", "max", "len", "length", "int", "fillna", "pow", "round", "sum", "all", "any",
            "is_increasing", "is_decreasing",
            "cumcount", "cummax", "cummin", "cumprod", "cumsum", "ngroup",
            "mean", "medium", "std", "sem", "prod", "sum", "count", "size", "groupby", "series_groupby", "apply",
            "if", "then", "elseif", "ifelse", "else", "ifthenelse",
            'lstrip', 'rstrip', 'strip', "concat", "padleft", "padright",
            'find', 'remove', 'slice', 'get', "str", "fillna", "rowid", "sort",
            "startswith", "endswith", "contains", "isin", "unique",
            "isnull", "notnull", "add", "sub", "mul", "div", "neg",
            "true", "True", "false", "False", "TRUE", "FALSE", "None", "Nan"]

pd_operations_1 = {'abs(': 'abs',
                   'isnull(': 'isnull',
                   'notnull(': 'notnull',
                   'any(': 'any',
                   'all(': 'all',
                   'mean(': 'mean',
                   'median(': 'median',
                   'std(': 'std',
                   'sem(': 'sem',
                   'min(': 'min',
                   'max(': 'max',
                   'sum(': 'sum',
                   'count(': 'count',
                   'size(': 'size',
                   'groupby(': 'groupby',
                   'series_groupby(': 'groupby',
                   }
pd_str_operations_1 = {'strip(': 'strip',
                       'lstrip(': 'lstrip',
                       'rstrip(': 'rstrip',
                       'length(': 'len',
                       }
pd_operations_2 = {'+': 'add',
                   'add': 'add',
                   '-': 'sub',
                   'sub': 'sub',
                   'subtract': 'sub',
                   '*': 'mul',
                   'mul': 'mul',
                   '/': 'div',
                   'div': 'div',
                   'pow(': 'pow',
                   'round(': 'round',
                   'isin(': 'isin',
                   'fillna(': 'fillna',
                   'sort(': 'sort_values',
                   '<': 'lt',
                   '<=': 'le',
                   '>': 'gt',
                   '>=': 'ge',
                   '==': 'eq',
                   '=': 'eq',
                   '!=': 'ne',
                   }
pd_str_operations_2 = {'concat(': 'cat',
                       'strip(': 'strip',
                       'lstrip(': 'lstrip',
                       'rstrip(': 'rstrip',
                       'slice(': 'slice',
                       'get(': 'get',
                       'startswith(': 'startswith',
                       'endswith(': 'endswith',
                       'contains(': 'contains',
                       'find(': 'find',
                       }
pd_operations_3 = {'padleft(': 'pad',
                   'padright(': 'pad'
                   }
pd_str_operations_3 = {'replace(': 'replace',
                       'slice(': 'slice',
                       }
pd_operations_n = {
    'unique(': 'unique',
    'isin(': 'isin',
    'max(': 'maxN',
    'min(': 'minN'
}

data_format_expression_map = {
    'apply_boolean': ChecksDataFormatExpression.BooleanExpression,
    'apply_numeric': ChecksDataFormatExpression.NumericExpression,
    'apply_string': ChecksDataFormatExpression.StringExpression,
    'apply_any': ChecksDataFormatExpression.AnyExpression,
}

expression_data_type_map = {
    ChecksDataFormatExpression.BooleanExpression: bool,
    ChecksDataFormatExpression.NumericExpression: float,
    ChecksDataFormatExpression.StringExpression: object,
    ChecksDataFormatExpression.AnyExpression: object,
}
expression_data_value_map = {
    ChecksDataFormatExpression.BooleanExpression: None,
    ChecksDataFormatExpression.NumericExpression: np.nan,
    ChecksDataFormatExpression.StringExpression: None,
    ChecksDataFormatExpression.AnyExpression: None,
}


def data_s(in_data) -> str:
    if isinstance(in_data, pd.Series):
        name = in_data.name
        if name:
            return str(name)
        return 'Series'
    if isinstance(in_data, pdgroup.DataFrameGroupBy):
        return 'DataFrameGroupBy'
    if isinstance(in_data, str):
        return f"'{in_data}'"
    return str(in_data)


class ChecksActions:
    def __init__(self, df: pd.DataFrame, has_data: bool = True) -> None:
        """
        Constructs the actionable object for the grammar that uses a stack of postfix data nodes.

        :param: df: DataFrame to run consistency checks.
        :param: has_data: Data is being calculated.
        """
        logging.debug(msg=f"Instantiate ChecksActions with df {df.shape} and has_data {has_data}.")
        self._df = df
        self._data: List[CheckDataPrimitive] = []
        self._postfix: List[str] = []
        self._has_data = has_data
        self._n_list = []
        self._expression = None

    @property
    def input_df(self) -> pd.DataFrame:
        """ The input Pandas DataFrame . """
        return self._df

    @property
    def data(self) -> List[CheckDataPrimitive]:
        """ The description of the data . """
        return self._data

    @property
    def postfix(self) -> List[str]:
        """
        The description of the data and operations in postfix form of the expression.

        :return: List of data and operation of expression.
        """
        return self._postfix

    @property
    def expression(self) -> Union[ChecksDataFormatExpression, None]:
        """ The expression access location."""
        return self._expression

    def operation_groupby(self, elements: List) -> None:
        """ Operation on group by data frames and series """
        logging.debug(msg=f"ChecksActions.operation_groupby(). elements {elements}")
        groupby_boolean_operation_1 = {'is_increasing(': 'is_monotonic_increasing',
                                       'is_decreasing(': 'is_monotonic_decreasing'}
        groupby_numeric_operation_1 = {'size(': 'size'}
        groupby_numeric_cum_operation_1 = {'ngroup(': 'ngroup'}
        groupby_series_boolean_operation_1 = {'any(': 'any',
                                              'all(': 'all'}
        groupby_series_numeric_operation_1 = {'mean(': 'mean',
                                              'median(': 'median',
                                              'std(': 'std',
                                              'sem(': 'sem',
                                              'min(': 'min',
                                              'max(': 'max',
                                              'prod(': 'prod',
                                              'sum(': 'sum',
                                              'count(': 'count'}
        groupby_series_numeric_cum_operation_1 = {'cumcount(': 'cumcount',
                                                  'cummax(': 'cummax',
                                                  'cummin(': 'cummin',
                                                  'cumprod(': 'cumprod',
                                                  'cumsum(': 'cumsum'}
        groupby_bool_operation_2 = {'any(': 'any',
                                    'all(': 'all'}
        groupby_numeric_operation_2 = {'mean(': 'mean',
                                       'median(': 'median',
                                       'std(': 'std',
                                       'sem(': 'sem',
                                       'min(': 'min',
                                       'max(': 'max',
                                       'prod(': 'prod',
                                       'sum(': 'sum',
                                       'count(': 'count'}
        groupby_numeric_cum_operation_2 = {'cumcount(': 'cumcount',
                                           'cummax(': 'cummax',
                                           'cummin(': 'cummin',
                                           'cumprod(': 'cumprod',
                                           'cumsum(': 'cumsum'}

        op = elements[0].text
        pd_groupby_op = None

        data_count = 1
        cumulative = False
        if elements[2].elements[0].text.startswith('series_groupby('):
            if groupby_boolean_operation_1.get(op):
                pd_groupby_op = groupby_boolean_operation_1.get(op)
                d_type = bool
            elif groupby_numeric_operation_1.get(op):
                pd_groupby_op = groupby_numeric_operation_1.get(op)
                d_type = float
            elif groupby_numeric_cum_operation_1.get(op):
                pd_groupby_op = groupby_numeric_cum_operation_1.get(op)
                d_type = float
                cumulative = True
            elif groupby_series_boolean_operation_1.get(op):
                pd_groupby_op = groupby_series_boolean_operation_1.get(op)
                d_type = bool
            elif groupby_series_numeric_operation_1.get(op):
                pd_groupby_op = groupby_series_numeric_operation_1.get(op)
                d_type = float
            elif groupby_series_numeric_cum_operation_1.get(op):
                pd_groupby_op = groupby_series_numeric_cum_operation_1.get(op)
                d_type = float
                cumulative = True
            else:
                msg = f"operation {op} is unsupported for Series GroupBy data."
                logging.error(msg)
                raise ValueError(msg)
        elif elements[2].elements[0].text.startswith('groupby('):
            if op == 'apply(':
                d_type = None
                data_count = 2
            elif groupby_numeric_operation_1.get(op):
                pd_groupby_op = groupby_numeric_operation_1.get(op)
                d_type = float
            elif groupby_numeric_cum_operation_1.get(op):
                pd_groupby_op = groupby_numeric_cum_operation_1.get(op)
                d_type = float
                cumulative = True
            elif groupby_bool_operation_2.get(op):
                pd_groupby_op = groupby_bool_operation_2.get(op)
                d_type = bool
                data_count = 2
            elif groupby_numeric_operation_2.get(op):
                pd_groupby_op = groupby_numeric_operation_2.get(op)
                d_type = float
                data_count = 2
            elif groupby_numeric_cum_operation_2.get(op):
                pd_groupby_op = groupby_numeric_cum_operation_2.get(op)
                d_type = float
                data_count = 2
                cumulative = True
            else:
                msg = f"Operation {op} is unsupported for DataFrame GroupBy data."
                logging.error(msg)
                raise ValueError(msg)
        else:
            msg = f"Operation {op} has unsupported text {elements[2].elements[0].text}."
            logging.error(msg)
            raise ValueError(msg)
        self._postfix.append(f"Operation: {op} on group by data")

        if len(self._data) < data_count:
            msg = f"Insufficient data stack size {len(self._data)} for {op} operation."
            logging.error(msg)
            raise ValueError(msg)
        d_2 = self._data.pop() if data_count >= 2 else None
        d_1 = self._data.pop()
        if not self._has_data:
            self._data.append(np.nan)
            return

        if not pd_groupby_op:
            if op == 'apply(':
                op_text = 'apply_any'
                if elements[5].elements[0].text == 'boolean_expr=':
                    op_text = 'apply_boolean'
                elif elements[5].elements[0].text == 'numeric_expr=':
                    op_text = 'apply_numeric'
                elif elements[5].elements[0].text == 'string_expr=':
                    op_text = 'apply_string'
                try:
                    if not isinstance(d_1, pdgroup.DataFrameGroupBy):
                        msg = f"Invalid data for DataFrameGroupBy {d_1}."
                        logging.error(msg)
                        raise ValueError(msg)
                    if not isinstance(d_2, str):
                        msg = f"Invalid data for DataFrameGroupBy {d_2}."
                        logging.error(msg)
                        raise ValueError(msg)
                    data_format_expression = data_format_expression_map.get(op_text)
                    d_type = expression_data_type_map[data_format_expression]
                    default_value = expression_data_value_map[data_format_expression]
                    r = None
                    for groupby_df in d_1.__iter__():
                        actions = ChecksActions(df=groupby_df[1])
                        obj_parser = ChecksParser(data_format_expression=data_format_expression,
                                                  input=d_2, actions=actions, types=None)
                        obj_parser.parse()
                        if len(actions.data) != 1:
                            msg = f"Invalid stack size {len(actions.data)} for result."
                            logging.error(msg)
                            raise ValueError(msg)
                        groupby_r = actions.data.pop()
                        if not isinstance(groupby_r, pd.Series):
                            if groupby_r is None or np.isnan(groupby_r):
                                continue
                            if groupby_df[1] is not None and not groupby_df[1].empty:
                                groupby_r = pd.Series(index=groupby_df[1].index, data=groupby_r)
                        if r is None:
                            if data_format_expression != actions.expression:
                                data_format_expression = actions.expression
                                d_type = expression_data_type_map[data_format_expression]
                                if d_type != groupby_r.dtype:
                                    if isinstance(groupby_r.dtype, bool):
                                        data_format_expression = ChecksDataFormatExpression.BooleanExpression
                                    elif isinstance(groupby_r.dtype, (int, float, complex, np.number)):
                                        data_format_expression = ChecksDataFormatExpression.NumericExpression
                                    elif np.issubdtype(groupby_r.dtype, np.integer):
                                        data_format_expression = ChecksDataFormatExpression.NumericExpression
                                    elif isinstance(groupby_r.dtype, str):
                                        data_format_expression = ChecksDataFormatExpression.StringExpression
                                    d_type = expression_data_type_map[data_format_expression]
                                default_value = expression_data_value_map[data_format_expression]
                            r = self.create_series(data=default_value, in_df=self._df, d_type=d_type)
                        r.update(groupby_r)
                except ParseError as e:
                    logging.error(f"Grammar parse error with expression '{d_2}'")
                    raise ParseError(e)
            else:
                msg = f"Operation {op} is unsupported for GroupBy data."
                logging.error(msg)
                raise ValueError(msg)
        else:
            gb_op = d_1.__getattribute__(pd_groupby_op)
            if inspect.ismethod(gb_op):
                gb1 = gb_op()
            elif isinstance(gb_op, pd.Series):
                gb1 = gb_op
            else:
                msg = f"Invalid data type {gb_op}."
                logging.error(msg)
                raise ValueError(msg)
            if data_count == 2 and isinstance(gb1, pd.DataFrame):
                if not isinstance(d_2, str):
                    msg = f"Invalid data for variable name {d_2}."
                    logging.error(msg)
                    raise ValueError(msg)
                gb_s = gb1[d_2]
            else:
                gb_s = gb1
            if cumulative:
                r = gb_s.copy()
            else:
                r = self.create_series(data=np.nan, in_df=self._df)
                for v, group in d_1:
                    s = pd.Series(gb_s.at[v], index=group.index.tolist())
                    r.update(s)
        r = r.astype(dtype=d_type)
        self._data.append(r)

    def operation_1(self, op: str) -> None:
        """
        Operation with single data retrieved from the data stack.

        :param: op: Operation name
        :return: None
        """
        logging.debug(msg=f"ChecksActions.operation_1(). op {op}")
        self._postfix.append(f"Operation: {op} using 1 data")
        if len(self._data) < 1:
            msg = f"Insufficient data stack size {len(self._data)} for {op} operation."
            logging.error(msg)
            raise ValueError(msg)
        d_1 = self._data.pop()
        if not self._has_data:
            self._data.append(np.nan)
            return

        pd_op = pd_operations_1.get(op)
        pd_str_op = pd_str_operations_1.get(op)
        logging.debug(f"Operation {op} with data {data_s(d_1)}")
        try:
            if not pd_op and not pd_str_op:
                if op == 'int(':
                    if isinstance(d_1, pd.Series):
                        r = d_1.astype(int)
                    elif isinstance(d_1, str):
                        r = int(d_1)
                    elif np.isnan(d_1):
                        r = np.nan
                    else:
                        r = int(d_1)
                elif op == 'not':
                    if isinstance(d_1, pd.Series):
                        r = ~d_1
                    elif np.isnan(d_1):
                        r = np.nan
                    else:
                        r = not d_1
                elif op == 'str(':
                    if isinstance(d_1, pd.Series):
                        r = d_1.astype(str)
                    elif isinstance(d_1, str):
                        r = d_1
                    elif np.isnan(d_1):
                        r = None
                    else:
                        r = str(d_1)
                elif op == '-':
                    if isinstance(d_1, pd.Series):
                        s = self.create_series(data=0, in_df=self._df)
                        r = s.sub(d_1)
                    elif np.isnan(d_1):
                        r = np.nan
                    else:
                        r = -d_1
                else:
                    msg = f"Operation {op} is unsupported format."
                    logging.error(msg)
                    raise ValueError(msg)
            elif pd_op == 'groupby':
                if op == 'series_groupby(':
                    if isinstance(d_1, pd.Series):
                        r = d_1.groupby(by=d_1.values)
                    else:
                        raise TypeError(f'operation {op} only supports series..')
                else:
                    if isinstance(d_1, pd.Series):
                        r = self._df.groupby(by=d_1.values)
                    elif isinstance(d_1, List):
                        r = self._df.groupby(by=d_1)
                    else:
                        raise TypeError(f'operation {op} only supports series or list of variable names.')
            elif isinstance(d_1, pd.Series):
                if pd_op == 'any':
                    r = True if pd.Series.any(d_1) else False
                elif pd_op == 'all':
                    r = True if pd.Series.all(d_1) else False
                else:
                    if pd_op:
                        r = d_1.__getattribute__(pd_op)
                    else:
                        r = d_1.str.__getattribute__(pd_str_op)
                    if callable(r):
                        r = r()
            elif isinstance(d_1, str):
                if pd_str_op == 'strip':
                    r = d_1.strip()
                elif pd_str_op == 'lstrip':
                    r = d_1.lstrip()
                elif pd_str_op == 'rstrip':
                    r = d_1.rstrip()
                elif pd_str_op == 'len':
                    r = len(d_1)
                elif pd_op == 'isnull':
                    r = False
                elif pd_op == 'notnull':
                    r = True
                else:
                    msg = f"Operation {op} is unsupported for string type."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, bool):
                if pd_op == 'isnull':
                    r = False
                elif pd_op == 'notnull':
                    r = True
                else:
                    msg = f"Operation {op} is unsupported for Boolean types."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, (int, float, np.number)):
                if pd_op == 'abs':
                    r = abs(d_1)
                elif pd_op == 'isnull':
                    r = True if np.isnan(d_1) else False
                elif pd_op == 'notnull':
                    r = False if np.isnan(d_1) else True
                elif pd_op == 'min':
                    r = d_1
                elif pd_op == 'max':
                    r = d_1
                elif op == '-':
                    r = -d_1
                elif np.isnan(d_1):
                    r = np.nan
                else:
                    msg = f"Operation {op} is unsupported for numeric types."
                    logging.error(msg)
                    raise ValueError(msg)
            elif d_1 is None:
                if pd_op == 'isnull':
                    r = True
                elif pd_op == 'notnull':
                    r = False
                else:
                    msg = f"Operation {op} is unsupported for None types."
                    logging.error(msg)
                    raise ValueError(msg)
            else:
                msg = f"Operation {op} is unsupported format."
                logging.error(msg)
                raise ValueError(msg)
        except TypeError as e:
            if self._df.empty:
                r = None if pd_str_op else np.nan
            else:
                raise TypeError(e)
        self._data.append(r)

    def operation_2(self, op: str) -> None:
        """
        Operation with two data operands retrieved from the data stack.

        :param: op: Operation name
        :return: None
        """
        logging.debug(msg=f"ChecksActions.operation_2(). op {op}")
        self._postfix.append(f"Operation: {op} using 2 data")
        if len(self._data) < 2:
            msg = f"Insufficient data stack size {len(self._data)} for {op} operation."
            logging.error(msg)
            raise ValueError(msg)
        d_2 = self._data.pop()
        d_1 = self._data.pop()
        if not self._has_data:
            self._data.append(np.nan)
            return

        pd_op = pd_operations_2.get(op)
        pd_str_op = pd_str_operations_2.get(op)
        logging.debug(f"Operation {op} with data {data_s(d_1)} and {data_s(d_2)}")
        try:
            if op == 'series_groupby(':
                if isinstance(d_1, pd.Series) and isinstance(d_2, pd.Series):
                    r = d_1.groupby(by=d_2.values)
                else:
                    raise TypeError(f'operation {op} only supports series..')
            elif isinstance(d_1, pd.Series):
                if pd_op:
                    if pd_op == 'sort_values':
                        r = d_1.sort_values(axis=0,  ascending=d_2, ignore_index=True)
                    else:
                        if op == '=':
                            logging.warning(f"Deprecated operation =. Use == instead.")
                        r = d_1.__getattribute__(pd_op)
                        if callable(r):
                            r = r(*[d_2])
                elif pd_str_op:
                    if pd_str_op == 'slice':
                        r = d_1.str.slice(d_2)
                    elif pd_str_op == 'get':
                        r = d_1.str.get(d_2)
                    else:
                        if pd_str_op == 'cat' and not isinstance(d_2, pd.Series):
                            d_2 = self.create_series(data=d_2, in_df=self._df)
                        r = d_1.str.__getattribute__(pd_str_op)
                        if callable(r):
                            r = r(*[d_2])
                elif op in ['and', 'or', 'xor']:
                    d_1.fillna(value=False, inplace=True)
                    if d_2 is None:
                        d_2 = self.create_series(data=False, in_df=self._df)
                    elif isinstance(d_2, pd.Series):
                        d_2.fillna(value=False, inplace=True)
                    if op == 'and':
                        r = d_1 & d_2
                    elif op == 'or':
                        r = d_1 | d_2
                    elif op == 'xor':
                        r = d_1 ^ d_2
                    else:
                        msg = f"Operation {op} is unsupported for Series data."
                        logging.error(msg)
                        raise ValueError(msg)
                elif op == 'concat(':
                    r = d_1 + d_2
                else:
                    msg = f"Operation {op} is unsupported for Series data."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_2, pd.Series):
                s = self.create_series(data=d_1, in_df=self._df)
                if pd_op:
                    if op == '=':
                        logging.warning(f"Deprecated operation =. Use == instead.")
                    r = s.__getattribute__(pd_op)
                    if callable(r):
                        r = r(*[d_2])
                elif pd_str_op:
                    r = s.str.__getattribute__(pd_str_op)
                    if callable(r):
                        r = r(*[d_2])
                elif op in ['and', 'or', 'xor']:
                    d_2.fillna(value=False, inplace=True)
                    if d_1 is None:
                        d_1 = self.create_series(data=False, in_df=self._df)
                    if op == 'and':
                        r = d_2 & d_1
                    elif op == 'or':
                        r = d_2 | d_1
                    elif op == 'xor':
                        r = d_2 ^ d_1
                    else:
                        msg = f"Operation {op} is unsupported for Series data."
                        logging.error(msg)
                        raise ValueError(msg)
                else:
                    msg = f"Operation {op} is unsupported for Series data."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, str):
                if pd_op == 'fillna':
                    r = d_1
                elif pd_op == 'isin':
                    if isinstance(d_2, List):
                        r = d_1 in d_2
                    else:
                        msg = f"Operation {op} is unsupported for string and non List type."
                        logging.error(msg)
                        raise ValueError(msg)
                elif pd_str_op == 'startswith':
                    r = d_1.startswith(d_2)
                elif pd_str_op == 'endswith':
                    r = d_1.endswith(d_2)
                elif pd_str_op == 'contains':
                    r = d_2 in d_1
                elif pd_str_op == 'cat':
                    r = d_1 + d_2
                elif pd_str_op == 'strip':
                    r = d_1.strip(d_2)
                elif pd_str_op == 'lstrip':
                    r = d_1.lstrip(d_2)
                elif pd_str_op == 'rstrip':
                    r = d_1.rstrip(d_2)
                elif pd_str_op == 'slice':
                    r = d_1[d_2:]
                elif pd_str_op == 'get':
                    r = d_1[d_2]
                elif pd_str_op == 'find':
                    r = d_1.find(d_2)
                elif op == '<=':
                    r = d_1 <= d_2
                elif op == '<':
                    r = d_1 < d_2
                elif op == '>=':
                    r = d_1 >= d_2
                elif op == '>':
                    r = d_1 > d_2
                elif op == '=':
                    logging.warning(f"Deprecated operation =. Use == instead.")
                    r = d_1 == d_2
                elif op == '==':
                    r = d_1 == d_2
                elif op == '!=':
                    r = d_1 != d_2
                elif op == 'concat(':
                    r = d_1 + d_2
                else:
                    msg = f"Operation {op} is unsupported for string type."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, bool):
                if d_2 is None:
                    d_2 = False
                if op == 'and':
                    r = d_1 & d_2
                elif op == 'or':
                    r = d_1 | d_2
                elif op == 'xor':
                    r = d_1 ^ d_2
                else:
                    msg = f"Operation {op} is unsupported for Boolean type."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, (int, float, np.number)):
                if pd_op == 'pow':
                    r = d_1 ** d_2
                elif pd_op == 'isin':
                    if isinstance(d_2, List):
                        r = d_1 in d_2
                    else:
                        msg = f"Operation {op} is unsupported for string and non List type."
                        logging.error(msg)
                        raise ValueError(msg)
                elif pd_op == 'round':
                    r = round(d_1, d_2)
                elif pd_op == 'fillna':
                    r = d_1 if not np.isnan(d_1) else d_2
                elif op == '+':
                    r = d_1 + d_2
                elif op == '-':
                    r = d_1 - d_2
                elif op == '*':
                    r = d_1 * d_2
                elif op == '/':
                    r = d_1 / d_2
                elif op == 'pow':
                    r = d_1 ** d_2
                elif op == '<=':
                    r = d_1 <= d_2
                elif op == '<':
                    r = d_1 < d_2
                elif op == '>=':
                    r = d_1 >= d_2
                elif op == '>':
                    r = d_1 > d_2
                elif op == '=':
                    logging.warning(f"Deprecated operation =. Use == instead.")
                    r = d_1 == d_2
                elif op == '==':
                    r = d_1 == d_2
                elif op == '!=':
                    r = d_1 != d_2
                elif np.isnan(d_1):
                    r = np.nan
                else:
                    msg = f"Operation {op} is unsupported for numeric types."
                    logging.error(msg)
                    raise ValueError(msg)
            elif d_1 is None:
                if pd_op == 'fillna':
                    r = d_2
                elif op == 'and':
                    r = False
                elif op == 'or':
                    r = d_2
                elif op == 'xor':
                    r = d_2
                elif op == '=' or op == '==':
                    r = True if d_2 is None else False
                elif op == '!=':
                    r = True if d_2 is not None else False
                else:
                    msg = f"Operation {op} is unsupported for None types."
                    logging.error(msg)
                    raise ValueError(msg)
            elif np.isnan(d_1):
                raise TypeError(f'operation {op} does not support nan.')
            else:
                msg = f"Operation {op} is unsupported format."
                logging.error(msg)
                raise ValueError(msg)
        except TypeError as e:
            if self._df.empty:
                r = None if pd_str_op else np.nan
            else:
                raise TypeError(e)
        self._data.append(r)

    def operation_3(self, op: str) -> None:
        """
        Operation with 3 data operands retrieved from the data stack.

        :param: op: Operation name
        :return: None
        """
        logging.debug(msg=f"ChecksActions.operation_3(). op {op}")
        self._postfix.append(f"Operation: {op} using 3 data")
        if len(self._data) < 3:
            msg = f"Insufficient data stack size {len(self._data)} for {op} operation."
            logging.error(msg)
            raise ValueError(msg)
        d_3 = self._data.pop()
        d_2 = self._data.pop()
        d_1 = self._data.pop()
        if not self._has_data:
            self._data.append(np.nan)
            return

        pd_op = pd_operations_3.get(op)
        pd_str_op = pd_str_operations_3.get(op)
        logging.debug(f"Operation {op} with data {data_s(d_1)}, {data_s(d_2)} and {data_s(d_3)}")
        try:
            if not pd_op and not pd_str_op:
                if op == 'ifthenelse(':
                    if isinstance(d_1, pd.Series):
                        if not isinstance(d_2, pd.Series):
                            d_2 = self.create_series(data=d_2, in_df=self._df)
                        if not isinstance(d_3, pd.Series):
                            d_3 = self.create_series(data=d_3, in_df=self._df)
                        r = d_3
                        r.loc[d_1] = d_2
                    elif isinstance(d_2, pd.Series) or isinstance(d_3, pd.Series):
                        if not isinstance(d_2, pd.Series):
                            d_2 = self.create_series(data=d_2, in_df=self._df)
                        if not isinstance(d_3, pd.Series):
                            d_3 = self.create_series(data=d_3, in_df=self._df)
                        r = d_2 if d_1 else d_3
                    else:
                        r = d_2 if d_1 else d_3
                else:
                    msg = f"Operation {op} is unsupported format."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, pd.Series):
                if pd_op == 'pad':
                    side = 'left' if op == 'padleft(' else 'right'
                    r = d_1.str.pad(width=d_2, side=side, fillchar=d_3)
                elif pd_str_op == 'replace':
                    r = d_1.str.replace(d_2, d_3)
                elif pd_str_op == 'slice':
                    r = d_1.str.slice(d_2, d_3)
                else:
                    msg = f"Operation {pd_op} is unsupported for series."
                    logging.error(msg)
                    raise ValueError(msg)
            elif isinstance(d_1, str):
                if op == 'padleft(':
                    r = d_1.rjust(d_2, d_3)
                elif op == 'padright(':
                    r = d_1.ljust(d_2, d_3)
                elif pd_str_op == 'replace':
                    r = d_1.replace(d_2, d_3)
                elif pd_str_op == 'slice':
                    r = d_1[d_2: d_3]
                else:
                    msg = f"Operation {pd_op} is unsupported for string."
                    logging.error(msg)
                    raise ValueError(msg)
            elif np.isnan(d_1):
                raise TypeError(f'operation {op} does not support nan.')
            else:
                msg = f"Operation {op} is unsupported format."
                logging.error(msg)
                raise ValueError(msg)
        except TypeError as e:
            if self._df.empty:
                r = np.nan
            else:
                raise TypeError(e)
        self._data.append(r)

    def operation_n(self, op: str) -> None:
        """
        Operation with ALL data operands retrieved from the data stack.

        :param: op: Operation name
        :return: None
        """
        logging.debug(msg=f"ChecksActions.operation_n(). op {op}")

        def concat_rows(row, features: pd.Index) -> str:
            return str([str(row[feat]) for feat in features])

        def get_unique(row, distribution: pd.Series) -> int:
            return distribution[row['target_data']]

        self._postfix.append(f"Operation: {op} using n data")

        data = []
        list_size = self._data.pop()
        for i in range(0, list_size):
            data.append(self._data.pop())
        if not self._has_data:
            self._data.append(np.nan)
            return
        data_strs = [data_s(d) for d in data]
        data_strs.reverse()
        logging.debug(f"Operation {op} with data {', '.join(data_strs)}")
        # print(op, "starting data:", data)
        try:
            pd_op = pd_operations_n.get(op)
            if not pd_op:
                msg = f"Operation {op} is unsupported format."
                logging.error(msg)
                raise ValueError(msg)
            elif pd_op == 'unique':
                # check if all are series
                for d in data:
                    if isinstance(d, pd.Series):
                        continue
                    elif isinstance(d, np.nan):
                        raise TypeError(f"Operation does not support {type(d)}")
                    else:
                        msg = f"Operation {op} is unsupported format."
                        logging.error(msg)
                        raise ValueError(msg)
                sub_df = pd.concat(data, axis=1)
                sub_df['target_data'] = sub_df.apply(lambda row: concat_rows(row, sub_df.columns), axis=1)
                dist = sub_df['target_data'].value_counts()
                r = sub_df.apply(lambda row: get_unique(row, dist), axis=1)
            elif pd_op == 'minN':
                constants = []
                sub_df = pd.DataFrame()
                for i, d in enumerate(data):
                    if isinstance(d, pd.Series):
                        sub_df[sub_df.shape[1]] = d
                    elif isinstance(d, np.int64) or isinstance(d, int):
                        constants.append(d)
                    elif isinstance(d, np.float64) or isinstance(d, float):
                        if np.isnan(d):
                            continue
                        constants.append(d)
                    else:
                        msg = f"Operation {op} is unsupported format."
                        logging.error(msg)
                        raise ValueError(msg)
                if not sub_df.empty:
                    if constants:
                        value = min(constants)
                        sub_df[sub_df.shape[1]] = self.create_series(data=value, in_df=sub_df)
                    r = sub_df.min(axis='columns')
                elif constants:
                    r = min(constants)
                else:
                    r = np.nan
            elif pd_op == 'maxN':
                constants = []
                sub_df = pd.DataFrame()
                for i, d in enumerate(data):
                    if isinstance(d, pd.Series):
                        sub_df[sub_df.shape[1]] = d
                    elif isinstance(d, np.int64) or isinstance(d, int):
                        constants.append(d)
                    elif isinstance(d, np.float64) or isinstance(d, float):
                        if np.isnan(d):
                            continue
                        constants.append(d)
                    else:
                        msg = f"Operation {op} is unsupported format."
                        logging.error(msg)
                        raise ValueError(msg)
                if not sub_df.empty:
                    if constants:
                        value = max(constants)
                        sub_df[sub_df.shape[1]] = self.create_series(data=value, in_df=sub_df)
                    r = sub_df.max(axis='columns')
                elif constants:
                    r = max(constants)
                else:
                    r = np.nan
            else:
                raise NotImplementedError(f"support for {op} or data has not been implemented.")
        except TypeError as e:
            if self._df.empty:
                r = np.nan
            else:
                raise TypeError(e)
        self._data.append(r)

    @staticmethod
    def create_series(data: CheckDataPrimitive, in_df: pd.DataFrame, d_type: pd.Series.dtype = None) -> pd.Series:
        if data is None:
            data_type = d_type
            if data_type is None:
                data_type = object
        elif isinstance(data, pd.Series):
            data_type = data.dtype
        elif isinstance(data, str) or isinstance(data, List):
            data_type = object
        elif isinstance(data, bool):
            data_type = bool
        elif isinstance(data, int):
            return pd.Series(data=data, index=in_df.index)
        else:
            data_type = float
        return pd.Series(data=data, index=in_df.index, dtype=data_type)

    def checks_expression(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Checks expression access location. """
        self._expression = ChecksDataFormatExpression.ChecksExpression
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def boolean_expression(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean expression access location. """
        self._expression = ChecksDataFormatExpression.BooleanExpression
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def numeric_expression(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric expression access location. """
        self._expression = ChecksDataFormatExpression.NumericExpression
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def string_expression(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String expression access location. """
        self._expression = ChecksDataFormatExpression.StringExpression
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def or_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean or, xor expression operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def and_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean and expression operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def not_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean not expression operation. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def isnull_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean is null function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def notnull_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean not null function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def starts_with_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String starts with function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def ends_with_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String ends with function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def contains_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String contains function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def is_in_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String is in function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def any_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean any function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def all_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean all function."""
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def numeric_comparison(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean numbers comparison operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def string_comparison(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean strings comparison operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def add_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric add, subtract expression operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def mul_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric multiple, divide expression operation. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def neg_expr(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric negative expression operation. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def abs_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric absolute function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def pow_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric power function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def round_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric round function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def length_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ The length of a string function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def find_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Find the lowest index corresponds to the position where the substring is fully contained. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def int_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Integer convert function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def fillna_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series fill not available function."""
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def rowid_func(self, _input: str, start: int, end: int) -> TreeNode:
        """ Series row identifier function. """
        logging.debug(f"Operation rowid")
        if self._df.empty:
            r = np.nan
        else:
            r = pd.Series(data=range(1, len(self._df) + 1))
        self._data.append(r)
        return TreeNode(text=_input[start: end], offset=start, elements=[])

    def sort_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series sort function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def mean_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series Mean function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def median_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series Median function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def std_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series standard deviation function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def sem_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series standard error over mean function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def min_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series minimum function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def max_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series maximum function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def sum_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series sum function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def count_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series count function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def size_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series size function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def unique_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Uniqueness function. """
        # Calculate the size of the parameters
        count = 1
        while self._n_list:
            cur_value = self._n_list.pop()
            if cur_value < start:
                self._n_list.append(cur_value)
                break
            count = count + 1

        self._postfix.append(f"Data count: {count}")
        self.data.append(count)
        self.operation_n(op=elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def max_list_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Maximum of multiple items in list function. """
        # Calculate the size of the parameters
        count = 1
        while self._n_list:
            cur_value = self._n_list.pop()
            if cur_value < start:
                self._n_list.append(cur_value)
                break
            count = count + 1

        self._postfix.append(f"Data count: {count}")
        self.data.append(count)
        self.operation_n(op=elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def min_list_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Minimum of multiple items in list function. """
        # Calculate the size of the parameters
        count = 1
        while self._n_list:
            cur_value = self._n_list.pop()
            if cur_value < start:
                self._n_list.append(cur_value)
                break
            count = count + 1

        self._postfix.append(f"Data count: {count}")
        self.data.append(count)
        self.operation_n(op=elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def next_value(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Next list value. """
        self._n_list.append(start)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def concat_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String concatenate function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def pad_left_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Pad left string function. """
        self.operation_3(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def pad_right_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Pad right string function. """
        self.operation_3(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def strip_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Strip string function. """
        if len(elements[3].elements) == 0:
            self.operation_1(elements[0].text)
        else:
            self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def left_strip_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Left strip string function. """
        if len(elements[3].elements) == 0:
            self.operation_1(elements[0].text)
        else:
            self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def right_strip_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Right strip string function. """
        if len(elements[3].elements) == 0:
            self.operation_1(elements[0].text)
        else:
            self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def replace_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Replace from a string. If the pattern is not present, the original string is returned. """
        self.operation_3(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def slice_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Slice substrings from the string data. """
        if len(elements[6].elements) == 0:
            self.operation_2(elements[0].text)
        else:
            self.operation_3(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def get_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Get element at position from a string. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def str_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String convert function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_any_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by any function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_all_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by all function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_is_increasing_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by is monotonic increasing function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_is_decreasing_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by is monotonic decreasing function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_cum_count_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by cum_count function.
        Number each item in each group from 0 to the length of that group - 1. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_cum_max_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by cum_max function. Cumulative maximum for each group. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_cum_min_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by cum_min function. Cumulative minimum for each group. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_cum_prod_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by cum_prod function. Cumulative product for each group. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_cum_sum_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by cum_sum function. Cumulative sum for each group. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_num_group_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by number group function. Number each group from 0 to the number of groups - 1. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_mean_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """Group by mean function. Compute mean of groups, excluding missing values. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_median_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by median function. Compute median of groups, excluding missing values. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_std_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by standard deviation function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_sem_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by standard error over mean function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_min_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by minimum function. Compute min of group values."""
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_max_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by maximum function. Compute max of group values. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_prod_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by product function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_sum_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by sum function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_count_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by count function. Compute count of group, excluding missing values. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_size_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by size function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_apply_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Group by apply function. """
        self.operation_groupby(elements=elements)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def groupby_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ DataFrame group by function. """
        self.operation_1(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def series_groupby_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Series group by function. """
        self.operation_2(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def ifthenelse_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ If Else function. """
        self.operation_3(elements[0].text)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def if_func(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ If function with support for then, elseif and else. """
        self._postfix.append(f"Operation: if then else using 3 data")
        if len(self._data) < 3:
            msg = f"Insufficient data stack size {len(self._data)} for if then else operation."
            logging.error(msg)
            raise ValueError(msg)
        r = self._data.pop()

        logging.debug(f"Operation if with else data {data_s(r)}")
        try:
            for i in range(len(elements[5].elements) + 1):
                if len(self._data) < 2:
                    msg = f"Insufficient data stack size {len(self._data)} for else if operation."
                    logging.error(msg)
                    raise ValueError(msg)
                d_2 = self._data.pop()
                d_1 = self._data.pop()
                logging.debug(f"Operation if data {data_s(d_1)} then data {data_s(d_2)}")

                if isinstance(d_1, pd.Series):
                    if not isinstance(d_2, pd.Series):
                        d_2 = self.create_series(data=d_2, in_df=self._df)
                    if not isinstance(r, pd.Series):
                        r = self.create_series(data=r, in_df=self._df)
                    r.loc[d_1] = d_2
                elif isinstance(d_2, pd.Series) or isinstance(r, pd.Series):
                    if not isinstance(d_2, pd.Series):
                        d_2 = self.create_series(data=d_2, in_df=self._df)
                    if not isinstance(r, pd.Series):
                        r = self.create_series(data=r, in_df=self._df)
                    r = d_2 if d_1 else r
                else:
                    r = d_2 if d_1 else r
        except TypeError as e:
            if self._df.empty:
                r = np.nan
            else:
                raise TypeError(e)
        self._data.append(r)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def list_value(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ List of values is added to the data stack. """
        # Calculate the size of the parameters
        count = 0
        if len(elements[2].elements) == 2:
            count += 1
            count += len(elements[2].elements[1].elements)
        list_values = [self.data.pop() for _ in range(count)]
        self._postfix.append(f"List Data size: {len(list_values)}")
        self.data.append(list_values)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def true_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean True data is added to the data stack. """
        self._postfix.append(f"Data: True")
        self._data.append(True)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def false_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Boolean False data is added to the data stack. """
        self._postfix.append(f"Data: False")
        self._data.append(False)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def none_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """None data is added to the data stack. """
        self._postfix.append(f"Data: None")
        self._data.append(None)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def nan_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """Nan data is added to the data stack. """
        self._postfix.append(f"Data: Nan")
        self._data.append(np.nan)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def integer_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric integer data is added to the data stack. """
        value = _input[elements[0].offset: elements[2].offset]
        self._postfix.append(f"Data integer: {value}")
        self._data.append(int(value))
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def float_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Numeric float data is added to the data stack. """
        value = _input[elements[1].offset: elements[4].offset]
        self._postfix.append(f"Data float: {value}")
        self._data.append(float(value))
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def expression_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Expression is added to the data stack. """
        value = self._data.pop()
        self._postfix.pop()
        self._postfix.append(f"{Postfix_Expression} {value}")
        self._data.append(value)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def string_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ String data is added to the data stack """
        value = elements[1].text
        self._postfix.append(f"Data string: {value}")
        self._data.append(value)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def variable(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Variable data is added to the data stack.
        The variable name is used to get the feature from the DataFrame. """
        value = _input[start: end]
        if value in keywords:
            return FAILURE
        self._postfix.append(f"{Postfix_Data_Variable} {value}")
        if not self._df.empty:
            self._data.append(self._df[value].copy())
        else:
            self._data.append(np.nan)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def variable_const(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ Variable name constance is added to the data stack. """
        value = elements[1].text
        if value in keywords:
            return FAILURE
        self._postfix.append(f"{Postfix_Data_Variable_Name} {value}")
        self._data.append(value)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)

    def variable_name_list(self, _input: str, start: int, end: int, elements: List) -> TreeNode:
        """ List of variable names."""
        # Calculate the size of the parameters
        count = 1
        if elements[3] is not None:
            if len(elements[3].elements) > 0:
                count += len(elements[3].elements)
        list_values = [self.data.pop() for _ in range(count)]
        self._postfix.append(f"Variable Name List Data size: {len(list_values)}")
        self.data.append(list_values)
        return TreeNode(text=_input[start: end], offset=start, elements=elements)
