from abc import ABC
from typing import Union, List, Dict
from pathlib import Path
from enum import Enum
from functools import total_ordering

import pandas as pd

ResultValueType = Union[int, float, str, List, Dict, pd.Series, pd.DataFrame]


@total_ordering
class ResultLevel(Enum):
    """The level of output for the report."""
    # Intermediate levels of results are not currently supported.
    SUMMARY = 1
    GENERAL = 2
    DETAIL = 4

    # The @total_ordering decorator requires at least one sorting comparison be defined.
    def __lt__(self, other) -> bool:
        """This determines if one ResultLevel is less than another."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Result(ABC):
    """
    A Result instance can take a number of different forms: int, list, mapping, etc.

    A Metric will output a list of Result instances.

    Each Report object will take a List[Result], which are obtained from a given set of Metrics that are run,
    and displays based on the List of Result instances passed in.
    """

    def __init__(self, value: ResultValueType, level: ResultLevel = ResultLevel.GENERAL,
                 metric_name: str = 'undefined', description: str = '', extra_info: str = '') -> None:
        """
        Constructor for Result.

        :param: value: value of the result.
        :param: level: Result level for the result. Default is SUMMARY.
        :param: metric_name: str of name for metric.
        :param: description: description of the result.
        :param: extra_info: extra information associated with the result. Default is ''.
        """

        self._value = value
        self._level = level
        self._description = description
        self._metric_name = metric_name
        self._extra_info = extra_info
        self._display_number_lines = None

    @property
    def value(self) -> ResultValueType:
        """Getter for value"""
        return self._value

    @property
    def level(self) -> ResultLevel:
        """Getter for  result level"""
        return self._level

    @property
    def metric_name(self) -> str:
        """Getter for metric_name"""
        return self._metric_name

    @metric_name.setter
    def metric_name(self, value: str):
        """Setter for encoder"""
        self._metric_name = value

    @property
    def description(self) -> str:
        """Getter for description"""
        return self._description

    @property
    def container(self) -> bool:
        """The result is a container object for other result objects."""
        return False

    def _container_test(self) -> bool:
        """Test the result is a container object for other result objects."""
        count = 0
        for v in self.value:
            if v is not None:
                if not isinstance(v, Result):
                    return False
                count += 1
        return count > 0

    def merge_result(self, result) -> None:
        """
        Merge two Results.

        :param: result: The Result to merge.
        :return: None
        """
        self._value = self._value + result.value

    def display(self) -> str:
        """The display string value for result."""
        return self._metric_description(self.display_value())

    def display_value(self) -> str:
        """The display string for the result's value."""
        return str(self._value)

    @property
    def display_number_lines(self) -> Union[int, None]:
        """display_number_lines of the report property."""
        return self._display_number_lines

    @display_number_lines.setter
    def display_number_lines(self, value: Union[int, None]) -> None:
        """Setter for the display_number_lines of the report property."""
        self._display_number_lines = value

    def _metric_description(self, value: str = '') -> str:
        """
        Metric description including the value and the extra information.

        :param: value: The value. Default = ''
        :return: String
        """
        output = f'Metric: {self.metric_name}, {self.description}: {value}'
        if self._extra_info:
            output += f'\n{self._extra_info}'
        return output

    def __str__(self) -> str:
        return self._metric_description(self.display_value())


class IntResult(Result):
    """Result subclass for values of type int"""

    def __init__(self, *args: int, **kwargs) -> None:
        """
        Constructor for IntResult

        :param: args: positional args
        :param: kwargs: keyword args
        """

        super().__init__(*args, **kwargs)


class FloatResult(Result):
    """Result subclass for values of type float"""

    def __init__(self, *args: float, **kwargs) -> None:
        """
        Constructor for FloatResult

        :param: args: positional args
        :param: kwargs: keyword args
        """

        super().__init__(*args, **kwargs)


class StrResult(Result):
    """Result subclass for values of type str"""

    def __init__(self, *args: str, **kwargs) -> None:
        """
        Constructor for StrResult

        :param: args: positional args
        :param: kwargs: keyword args
        """

        super().__init__(*args, **kwargs)

    def merge_result(self, result: Result) -> None:
        """
        Merge two StrResults.

        :param: result: The StrResult to merge.
        :return: None
        """
        if not isinstance(result, StrResult):
            raise ValueError(f'result is not of type StrResult')

        if self.metric_name == "Synthesized features":
            if self._value != "Undefined":
                if result.value == "Undefined":
                    self._value = "Undefined"
                if self._value != result.value:
                    self._value = "Undefined"
            return

        self._value = self._value + result.value


class IndexResult(Result):
    """Result subclass for values of list of indexes."""
    def __init__(self, *args: List[int], **kwargs) -> None:
        """
        Constructor for IndexResult

        :param: args: positional args
        :param: kwargs: keyword args
        """
        super().__init__(*args, **kwargs)

    def display_value(self) -> str:
        """The display string for the result's value."""
        output: str = '\n Indexes:'
        output += ' '.join([f' {str(item)}' for item in self.value])
        return output


class ListResult(Result):
    """Result subclass for values of type List"""

    def __init__(self, *args: List, **kwargs) -> None:
        """
        Constructor for ListResult

        :param: args: positional args
        :param: kwargs: keyword args
        """
        super().__init__(*args, **kwargs)

    @property
    def container(self) -> bool:
        """The result is a container object for other result objects."""
        return self._container_test()

    def display_value(self) -> str:
        """The display string for the result's value."""
        output: str = '\n'
        if self.display_number_lines is not None and self.display_number_lines > 0:
            output += '\n'.join([str(item) for item in self.value[:self.display_number_lines]])
        else:
            output += '\n'.join([str(item) for item in self.value])
        return output

    def save(self, save_dir: str) -> None:
        """
        Save full detail of list result to a file.

        :param: save_dir: string specifying path to folder to save the file
        """
        save_path = Path(save_dir)
        if not save_path.exists() or not save_path.is_dir():
            return
        file_path = save_path.joinpath(f'{self.description}.txt')
        with open(file_path, 'w') as open_file:
            output: str = '\n'
            output += '\n'.join([str(item) for item in self.value])
            open_file.write(output)


class MappingResult(Result):
    """Result subclass for values of type Dict"""

    def __init__(self, *args: Dict, **kwargs) -> None:
        """
        Constructor for MappingResult

        :param: args: positional args
        :param: kwargs: keyword args
        """
        super().__init__(*args, **kwargs)

    @property
    def container(self) -> bool:
        """The result is a container object for other result objects."""
        return self._container_test()

    def display_value(self) -> str:
        """The display string for the result's value."""
        def _is_list_container(value) -> bool:
            for item in value:
                if isinstance(item, Dict):
                    return True
            return False

        def _display_str(value) -> str:
            nonlocal tab_size
            if isinstance(value, List) and _is_list_container(value):
                tab_size += 4
                sep = '\n' + ' ' * tab_size
                list_out = '[' + sep + sep.join([f'{_display_str(item)}' for item in value]) + ']'
                tab_size -= 4
                return list_out
            elif isinstance(value, Dict):
                tab_size += 4
                sep = '\n' + ' ' * tab_size
                dict_out = '{' + sep + sep.join([f'{k}: {_display_str(v)}' for k, v in value.items()]) + '}'
                tab_size -= 4
                return dict_out
            return str(value)

        tab_size = 0
        output: str = '\n'
        if self.display_number_lines is not None and self.display_number_lines > 0:
            output += '\n'.join([f'{key}: {_display_str(val)}' for key, val in
                                 list(self.value.items())[:self.display_number_lines]])
        else:
            output += '\n'.join([f'{key}: {_display_str(val)}' for key, val in self.value.items()])
        return output


class PandasResult(Result):
    """Result subclass for Pandas objects"""

    def __init__(self, *args: Union[pd.Series, pd.DataFrame], **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def display_value(self) -> str:
        """The display string for the result's value."""
        output: str = '\n'
        output += self.value.__str__()
        return output


class PlotResult(Result):
    """
    THIS IS WIP

    """
    # TODO: Integrate this with the evaluate pipeline, and then change TableMetrics compute_result to
    #  also create/return PlotResults
    # The main obstacle will be changing the reporting framework to save pngs as external files

    def __init__(self, function, file_name: str,  *args, function_dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.function = function
        self.function_dict = function_dict
        self.file_name = file_name

    def _metric_description(self, value: str = '') -> str:
        pass
    
    def save(self, save_dir: str) -> None:
        """
        Save full detail for plot result

        :param: save_dir: string specifying path to folder to save the file
        """
        save_path = Path(save_dir)
        if not save_path.exists() or not save_path.is_dir():
            return
        self.function(**self.function_dict, results=self.value, filepath=save_path / self.file_name)

    def __str__(self) -> str:
        return self._metric_description("PlotResult Saved.")


class ParquetResult(Result):
    """
    Hack for intercepting per-value-tuple marginals and saving them to a parquet file
    """
    def __init__(self, *args: Union[pd.Series, pd.DataFrame], **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f'Metric: {self.metric_name}\n'

    def display_value(self) -> str:
        """The display string for the result's value."""
        output: str = '\n'
        output += self.value.__str__()
        return output

    def save(self, save_dir: str) -> None:
        """
        Save full detail marginals to individual parquet files within the folder specified by parquet_path

        :param: save_dir: string specifying path to folder to save the file
        """
        save_path = Path(save_dir)
        if not save_path.exists() or not save_path.is_dir():
            return

        # self._value will be a dictionary of dataframes, keyed by marginal
        # Dataframe saved to parquet
        for marginal, frame in self._value.items():
            marginal_file_name = ('_'.join(marginal)) + '.parquet'
            frame.to_parquet(path=save_path / marginal_file_name)


class BooleanResult(Result):
    """Result subclass for values of type boolean"""

    def __init__(self, *args: int, **kwargs) -> None:
        """
        Constructor for BooleanResult

        :param: args: positional args
        :param: kwargs: keyword args
        """

        super().__init__(*args, **kwargs)


class AssertBooleanResult(Result):
    """Result subclass for values of type boolean"""

    def __init__(self, expected_value, *args: int, **kwargs) -> None:
        """
        Constructor for BooleanResult

        :param: args: positional args
        :param: kwargs: keyword args
        """
        self._expected_value = expected_value
        super().__init__(*args, **kwargs)

    @property
    def expected_value(self) -> bool:
        return self._expected_value

    # def _metric_description(self, value: str = '') -> str:
    #     """
    #     Metric description including the value and the extra information.
    #
    #     :param: value: The value. Default = ''
    #     :return: String
    #     """
    #     output = f'Metric: {self.metric_name}, {self.description}: {value}'
    #     if self._extra_info:
    #         output += f'\n{self._extra_info}'
    #     return output

    def __str__(self) -> str:
        return self._metric_description(str(self.value))
