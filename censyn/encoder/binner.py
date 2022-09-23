import itertools
import logging
import math
import re
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Tuple, Any, Union, Iterable

import numpy as np
import pandas as pd


# Constant indicating the index for null/none/nan values
NULL_INDEX:  int = 0
NULL_TEXT: str = 'Null'
# Constant indicating the index for non-null values that don't fit in any provided bins
OTHER_INDEX: int = 1
OTHER_TEXT: str = 'Other'
SMALL_SIZE: float = 0.000001


class BinnerExpandType(Enum):
    """
    This is an enum that represents the type of expansion for a binning tuple
    """
    linear = 0
    logarithmic = 1
    percentile = 2


def percentile_format(value: int) -> str:
    """
    Create format string for the percentile values.

    :param: value: Number of percentile bins.
    :return: Format string
    """
    if value in [1, 2, 4, 5, 10, 20, 25, 50, 100]:
        return ".0f"
    if value <= 50:
        return ".1f"
    if value <= 200:
        return ".2f"
    if value <= 1000:
        return ".3f"
    return ""


def to_num(s: str):
    """
    Turn a string into a number.
    Everything is either an int, float, or np.inf
    """
    try:
        if re.fullmatch(r'-?\d+', s):
            return int(s)
        return float(s)
    except ValueError as e:
        if re.fullmatch(r'np\.inf', s):
            return np.inf
        elif re.fullmatch(r'-np\.inf', s):
            return -np.inf
        raise ValueError(e)


def pairwise_iterable(iterable) -> Iterable:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def indexed_pairwise_mapping(iterable) -> Dict:
    """s -> {0:(s0,s1), 1:(s1,s2), 2:(s2, s3), ...}"""
    return {index: pair for index, pair in enumerate(pairwise_iterable(iter(iterable)))}


def valid_number_bins(value: Union[np.number, str]) -> int:
    """
    The number of bins must be a positive integer value. Convert type as needed.

    :param value: Numeric value.
    :return: Number of bins as integer.
    :raise: ValueError if not a valid positive integer.
    """
    if isinstance(value, str):
        value = to_num(value)
    if not isinstance(value, int):
        if value == math.inf or value == -math.inf:
            msg = f"Invalid bin format {value}. {value} must be a positive integer."
            logging.error(msg=msg)
            raise ValueError(msg)
        value_int = int(round(value))
        if abs(value - value_int) > SMALL_SIZE:
            msg = f"Invalid bin format {value}. {value} must be a positive integer."
            logging.error(msg=msg)
            raise ValueError(msg)
        value = value_int
    if value <= 0:
        msg = f"Invalid bin format {value}. {value} must be a positive integer."
        logging.error(msg=msg)
        raise ValueError(msg)
    return value


def parse_tuple(value, is_numeric: bool) -> tuple:
    """
    Takes a string in the format of a tuple of ints or floats, parses it, and returns it as a tuple.
    Within the tuple string, np.inf and -np.inf parse respectively. All other non-numerical values throw an error.

    Any non-string, or string not formatted as a tuple, is returned without any conversion.

    :param: is_numeric: Boolean for the data is numeric
    :param: value: A string in the format of a tuple, or any other value.
    :return: The parsed tuple, if the value is a string in tuple format. Otherwise, the original value.
    :raise: ValueError if value can not be parsed.
    """
    if not isinstance(value, str) or not re.fullmatch(r'\(.*\)', value):
        return value

    parsed: list = re.findall(r'^\( *([\w\d.-]+) *\)$', value)
    if not parsed:
        parsed = re.findall(r'^\( *([\w\d.-]+), *([\w\d.-]+) *\)$', value)
    if not parsed:
        parsed = re.findall(r'^\( *([\w\d.-]+), *([\w\d.-]+), *([\w\d.-]+) *\)$', value)

    if not parsed:
        raise ValueError('Invalid format', value)
    if isinstance(parsed[0], str):
        parsed[0] = [parsed[0]]

    # Set the items.
    if is_numeric:
        items = [to_num(i) for i in parsed[0]]
    else:
        items = [i for i in parsed[0]]

    # Validate the items of the tuple.
    if len(items) == 1:
        items[0] = valid_number_bins(items[0])
    if len(items) >= 2:
        if items[1] <= items[0]:
            msg = f"Invalid bin format {value}. {items[1]} must be larger than {items[0]}."
            logging.error(msg=msg)
            raise ValueError(msg)
    if len(items) == 3:
        items[2] = valid_number_bins(items[2])
    return tuple(items)


def expand_space_tuple(value: tuple, expand_type: BinnerExpandType) -> Union[List, tuple]:
    """
    A 3-tuple is used to indicate a range of values that will be linearly discrete.
    This function will convert any 3-tuple into a corresponding set of 2-tuples, where each 2-tuple indicates
    a range of values, and return them in a list.

    Any other tuple passed in is just returned.

    For 3-tuple (X,Y,Z):
    - The returned list be of size Z
    - Each 2-tuple will be of equal size
    - The set of 2-tuples will cover the entire interval between X and Y

    For example:
    (0, 10, 2) will return [(0, 5), (6, 10)]
    (0.1, 0.3, 3) will return [(0.1, 0.2), (0.2, 0.3), (0.1, 0.2)]

    :param: value: A 3-tuple, or any other value.
    :return: If value is a 3-tuple, a list of equal sized 2-tuples covering the intervals indicated by value.
             Otherwise, value.
    """
    # Only tuples of length 3 need to be expanded
    if len(value) != 3:
        return value

    value_0 = value[0]
    if isinstance(value_0, str):
        value_0 = to_num(value_0)
    value_1 = value[1]
    if isinstance(value_1, str):
        value_1 = to_num(value_1)

    # Create the binning intervals
    if expand_type == BinnerExpandType.linear:
        intervals = np.linspace(value_0, value_1, num=value[2] + 1)
        interval_size = intervals[1] - intervals[0]
    elif expand_type == BinnerExpandType.logarithmic:
        intervals = np.geomspace(value_0, value_1, num=value[2] + 1)
        interval_size = min(intervals[1] - intervals[0], intervals[-1] - intervals[-2])
    elif expand_type == BinnerExpandType.percentile:
        return [value]
    else:
        msg = f"Invalid BinnerExpandType {expand_type}."
        logging.error(msg)
        raise ValueError(msg)

    # Very small interval so no rounding
    if interval_size < SMALL_SIZE:
        return [pair for pair in pairwise_iterable(iter(intervals))]

    # Calculate rounding precision.
    round_precision = 1
    while interval_size < 1.0:
        round_precision = round_precision + 1
        interval_size = interval_size * 10

    def _round_pair(pair) -> tuple:
        return round(pair[0], round_precision), round(pair[1], round_precision)

    return [_round_pair(pair) for pair in pairwise_iterable(iter(intervals))]


class Binner:
    """
    Binning provides a way to map a Feature's values to a specific and discrete set of values.
    Some uses of binning include:
    - Reduce the cardinality of a high cardinality feature
    - Map a continuous feature to a discrete set of intervals
    - Map a set of categorical labels to a parallel set of numerical values

    Binning also provides a method for mapping values from their binned values to their human-readable labels.

    Specifics on valid mapping formats are described in clean_bin_mapping.
    """

    def __init__(self, is_numeric: bool, mapping: Dict) -> None:
        """
        Initializer for the Binner.
        This will clean the provided bin mapping, as described in clean_bin_mapping.

        :param: is_numeric: Boolean for the data is numeric
        :param mapping: The bin mapping.
        """
        logging.debug(msg=f"Instantiate Binner. is_numeric {is_numeric} and mapping {mapping}.")
        self._is_numeric: bool = is_numeric
        self._mapping = mapping
        self._bin_list = None

    @property
    def bin_list(self) -> Union[List[Tuple[str, Any]], None]:
        """
        The list of bins.
        Each item in the list is a 2-tuple (label, condition), where:
         - the condition indicates which value(s) fit into the bin
         - the label is the human-readable string for the bin
        """
        return self._bin_list

    def create_bins(self, in_s: Union[pd.Series, None] = None) -> None:
        """
        Create the bin list.

        :param: in_s: The input Pandas Series data to bin.
        :return: None
        """
        if len(self._mapping) > 0:
            # Initializing the list to be able to set the special indices directly
            self._bin_list: List[Tuple[str, Any]] = [('', None)]*2
            self._bin_list[NULL_INDEX] = (NULL_TEXT, NULL_TEXT)
            self._bin_list[OTHER_INDEX] = (OTHER_TEXT, OTHER_TEXT)

            clean_mapping = Binner.clean_bin_mapping(self._is_numeric, self._mapping, in_s=in_s)

            self._bin_list.extend([(key, value) for (key, value) in clean_mapping.items()])
        else:
            self._bin_list: List[Tuple[str, Any]] = []

    @staticmethod
    def clean_bin_mapping(is_numeric: bool, mapping: Dict, in_s: Union[pd.Series, None] = None) -> Dict:
        """
        Take a mapping configuration and return a new cleaned mapping.
        For most values in the mapping, this function uses and returns them as provided.
        For all tuple-formatted strings, this function converts and/or expands them, as follows:

        Numerical 2-tuples are used to indicate a range of values, and are converted directly.

        For example:
        'X': '(1, 2)' is converted to 'X' : (1, 2)
        'X': '(-123, 50)' is converted to 'X' : (-123, 50)

        np.inf and -np.inf are supported:
        'X': '(-np.inf, np.inf)' is converted to 'X' : (-np.inf, np.inf)

        The first value must be less than the second.
        'X': '(50, 10)' will raise a ValueError.

        Numerical 3-tuples are used to indicate a linear discretion of values covering a range.
        In practice, this means that the value range (as indicated by the first two values)
        is evenly split into the desired number of bins (as indicated by the third value).
        New 2-tuple ranges are added to the mapping to indicate the ranges covered by each of these bins.
        The keys are modified to indicate the exact range of each bin.

        For example:
        'X': '(0, 10, 2)' is converted to
            'X [0, 5)' : (0, 5)
            'X [6, 10)' : (6, 10)

        'X': '(-100, 0, 100)' is converted to
            'X [-100, -99)' : (-100, -99)
            'X [-99, -98)' : (-99, -98)
                    ... (97 more tuples)
            'X [-1, 0)' : (-1, 0)

        As above, the first value must be less than the second.
        'X': '(50, 10, 2)' is invalid and will raise a ValueError.

        The third value (indicating the number of bins) must be positive.
        'X': '(1, 10, 0)' will raise a ValueError.

        :param: is_numeric: Boolean for the data is numeric
        :param: mapping: The original mapping
        :param: in_s: The input Pandas Series data to bin.
        :return: The cleaned mapping
        :raises: ValueError if any invalid tuple strings are provided.
        """
        logging.debug(msg=f"Binner.clean_bin_mapping(). is_numeric {is_numeric} and mapping {mapping}.")

        cleaned_mapping = {}
        for key, value in mapping.items():
            # Map of linear tuple bins
            parsed_linear = parse_tuple(value=value, is_numeric=is_numeric)
            if isinstance(parsed_linear, tuple):
                Binner._validate_tuple(key, parsed_linear)

                if len(parsed_linear) == 2:
                    # A valid tuple of length 2 can just be used directly
                    cleaned_mapping[key] = parsed_linear
                    continue

                # Handle a tuple of length 3, indicating a linear discretion,
                # by creating the appropriate set of tuples of length 2
                cur_map = {f'{key} [{str(pair[0])} {str(pair[1])})': pair
                           for pair in expand_space_tuple(value=parsed_linear, expand_type=BinnerExpandType.linear)}
                cleaned_mapping.update(cur_map)
                continue

            # Map of logarithmic tuple bins.
            if isinstance(value, str) and value.startswith("log"):
                parsed_log = parse_tuple(value=value[3:], is_numeric=is_numeric)
                if isinstance(parsed_log, tuple) and len(parsed_log) == 3:
                    Binner._validate_log_tuple(key, parsed_log)

                    # Handle a tuple of length 3, indicating a log discretion,
                    # by creating the appropriate set of tuples of length 2
                    cur_map = {f'{key} [{str(pair[0])} {str(pair[1])})': pair
                               for pair in expand_space_tuple(value=parsed_log,
                                                              expand_type=BinnerExpandType.logarithmic)}
                    cleaned_mapping.update(cur_map)
                    continue

            # Map of percentile tuple bins.
            if isinstance(value, str) and value.startswith("percentile"):
                parsed_log = parse_tuple(value=value[10:], is_numeric=is_numeric)
                if isinstance(parsed_log, tuple) and len(parsed_log) in [1, 3]:
                    def _percent_d(p) -> str:
                        nonlocal percent_format
                        return f"{p * 100:{percent_format}}%"
                    if in_s is None or in_s.empty:
                        msg = f"Series must have valid values for percentile binning."
                        logging.error(msg)
                        raise ValueError(msg)
                    parsed_value = parsed_log[0] if len(parsed_log) == 1 else parsed_log[2]
                    percent_format = percentile_format(parsed_value)
                    q = [0.0]
                    q.extend([1.0 * (i + 1) / parsed_value for i in range(parsed_value)])
                    if len(parsed_log) == 3:
                        in_s = in_s[in_s >= parsed_log[0]]
                        in_s = in_s[in_s < parsed_log[1]]
                    q_values = in_s.quantile(q).tolist()
                    cur_map = OrderedDict()
                    for i in range(1, len(q_values)):
                        cur_map[f"{key} {_percent_d(q[i - 1])} - {_percent_d(q[i])}: "
                                f"{q_values[i - 1]} - {q_values[i]}"] = (q_values[i - 1], q_values[i])
                    last_item = list(cur_map.items())[-1]
                    if len(parsed_log) == 1:
                        last_value = [last_item[1][0], last_item[1][1] + SMALL_SIZE]
                    else:
                        last_value = [last_item[1][0], parsed_log[1]]
                    cur_map[last_item[0]] = tuple(last_value)
                    cleaned_mapping.update(cur_map)
                    continue

            # It is not a tuple, just use the value as entered
            cleaned_mapping[key] = value
            continue

        return cleaned_mapping

    @staticmethod
    def _validate_log_tuple(key: str, value: tuple) -> None:
        Binner._validate_tuple(key=key, value=value)

        # Values must not equal zero.
        if value[0] == 0 or value[1] == 0:
            msg = f"{key} is an invalid value. The values must not be 0."
            logging.error(msg=msg)
            raise ValueError(msg)

        # Both values must be either positive or negative.
        if value[0] * value[1] <= 0:
            msg = f"{key} is an invalid value. Both values must be either positive or negative."
            logging.error(msg=msg)
            raise ValueError(msg)

    @staticmethod
    def _validate_tuple(key: str, value: tuple) -> None:
        """
        Function for validating handled tuples. Valid tuples follow these rules:
        - Tuples must be of length 2 or 3 only
        - The first value in the tuple must be less than the second
        - The third value, if it exists, must be positive

        :param: key: Key indicating the key of the tuple to provide helpful error feedback
        :param: value: Tuple to be checked for validity
        :return: None
        :raises: ValueError if any of the above rules are invalidated
        """

        if not len(value) in [2, 3]:
            raise ValueError(f'{key} is an invalid mapping. Only tuples of length 2 or 3 are allowed.')

        # It is a tuple of length 2 or 3. Ensure the first value is less than the second
        if value[0] >= value[1]:
            raise ValueError(f'{key} is an invalid range mapping. The first value must be less than the second')

        # Tuples of length 3 must have a positive third value, but not infinity
        if len(value) == 3 and (value[2] <= 0 or value[2] == math.inf):
            raise ValueError(f'{key} is an invalid linear discretion mapping. '
                             f'The third value must be positive and not infinite')

    def bin(self, in_s: pd.Series) -> pd.Series:
        """
        Method to apply bins to data. Each value in the input DataFrame will be converted to the
        corresponding bin into which that it fits, and a new Series with the bin indices

        A value of OTHER_INDEX indicates that the value did not fit into any of the supplied bins.
        A value of NULL_INDEX indicates that the value is None or NaN.
        This convert the values in the given DataFrame,
        returning the indices corresponding to the bins from the bin list.

        :param: in_s: The input Pandas Series data to bin.
        :return: A new Series with the resulting binned data.
        """
        logging.debug(msg=f"Binner.bin(). in_s {in_s.shape}.")

        if self.bin_list is None:
            self.create_bins(in_s=in_s)

        bin_s = in_s.copy() if len(self.bin_list) == 0 else pd.Series(data=OTHER_INDEX, dtype=int, index=in_s.index)

        # Create null bin
        null_mask = (in_s.isnull())
        if np.issubdtype(in_s.dtype, np.number):
            null_mask = null_mask | (in_s.isna())
        if any(null_mask):
            bin_s.loc[null_mask] = NULL_INDEX

        # Fill each data bin.
        for index, (key, value) in enumerate(self.bin_list):
            # mask is True when original variable value is in bin.
            if key == NULL_TEXT or key == OTHER_TEXT:
                continue

            # Calculate mask for the bin
            try:
                if isinstance(value, tuple):
                    if np.issubdtype(in_s.dtype, np.number):
                        mask = (value[0] <= in_s) & (in_s < value[1])
                    else:
                        mask = (str(value[0]) <= in_s) & (in_s < str(value[1]))
                elif isinstance(value, list):
                    mask = (in_s.isin(value))
                elif not np.issubdtype(type(value), np.number) and np.issubdtype(in_s.dtype, np.number):
                    mask = (in_s == int(value))
                else:
                    mask = (in_s == value)
                if any(mask):
                    bin_s.loc[mask] = index
            except ValueError:
                logging.error(f"Invalid bin value {value}.")

        return bin_s

    def bins_to_labels(self, in_s: pd.Series) -> pd.Series:
        """
        Method to convert binned values back to their labels

        :param: in_s: The binned, Pandas Series data.
        :return: A new Series with the labels for each of the binned values.
        """
        m = {index: key for (index, (key, value)) in enumerate(self.bin_list)}
        return in_s.map(m)
