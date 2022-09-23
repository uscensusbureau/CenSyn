import logging
import re
from typing import Dict, List, Union

from .checks import keywords
from .checks_data_calculator import AnyDataCalculator


def rename_variables(expression: Union[str, List[str]], rename: Dict[str, str], validate: bool = False) -> str:
    """
    Rename the variables in a Data Calculate Grammar Expression.

    :param: expression: String expression or list of string to compose an expression.
    :param: replace: Dictionary of old and new variable names.
    :param: validate: Boolean flag to validate the rename-dictionary.
    :return: The modified expression.
    """
    logging.debug(msg=f"checks_utils.rename_variables(). expression {expression}, rename {rename} and "
                      f"validate {validate}.")
    
    def _validate_dict() -> None:
        for old_v, new_v in rename.items():
            if not isinstance(old_v, str):
                msg = f"Error: Invalid data type for rename dictionary key {old_v}."
                logging.error(msg)
                raise ValueError(msg)
            if not isinstance(new_v, str):
                msg = f"Error: Invalid data type for rename dictionary value {new_v}."
                logging.error(msg)
                raise ValueError(msg)
            if old_v in keywords:
                msg = f"Error: Cannot use keyword {old_v} for rename dictionary key."
                logging.error(msg)
                raise ValueError(msg)
            if new_v in keywords:
                msg = f"Error: Cannot use keyword {new_v} for rename dictionary value."
                logging.error(msg)
                raise ValueError(msg)
            if (len(old_v) == 0 or not (old_v[0].isalpha() or old_v[0] == '_') or
                    re.search(r'[^a-zA-Z0-9_]', old_v)):
                msg = f"Error: Invalid data value for rename dictionary key {old_v}."
                logging.error(msg)
                raise ValueError(msg)
            if (len(new_v) == 0 or not (new_v[0].isalpha() or new_v[0] == '_') or
                    re.search(r'[^a-zA-Z0-9_]', new_v)):
                msg = f"Error: Invalid data value for rename dictionary value {new_v}."
                logging.error(msg)
                raise ValueError(msg)

    # Validate the rename-dictionary
    if validate:
        _validate_dict()

    # Initialize the output expression
    out_expr = " ".join(expression) if isinstance(expression, List) else expression

    if out_expr:
        # Process the variable in the expression.
        variables = AnyDataCalculator(feature_name='test', expression=out_expr).compute_variables()
        for var in variables:
            if var not in rename.keys():
                continue
            len_expr = len(out_expr)
            start_i = 0
            while start_i < len_expr:
                start_i = out_expr.find(var, start_i)
                if start_i < 0:
                    break
                if start_i > 0 and (out_expr[start_i - 1].isalnum() or out_expr[start_i - 1] == '_'):
                    start_i += len(var)
                    continue
                if start_i + len(var) < len_expr and (out_expr[start_i + len(var)].isalnum() or
                                                      out_expr[start_i + len(var)] == '_'):
                    start_i += len(var)
                    continue
                logging.debug(msg=f"Rename variable {var} to {rename[var]} at {start_i} in "
                                  f"expression '{out_expr[0:60]}'")
                out_expr = out_expr[:start_i] + rename[var] + out_expr[start_i + len(var):]
                start_i += len(rename[var])

    return out_expr
