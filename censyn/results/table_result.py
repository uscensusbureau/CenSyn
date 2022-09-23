import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from .result import Result, PandasResult


class TableResult(PandasResult):
    """Result subclass for Table objects"""

    def __init__(self, *args, sort_column: str = None, ascending: bool = True, factor: float = 1.0, **kwargs) -> None:
        """
        Initialize the Table Result.

        :param: args: Positional arguments passed on to super.
        :param: sort_column: The name of the column for sorting of the table.
        :param: ascending: The sort values are ascending. Default value is True.
        :param: factor: The factor of the table. Default value is 1.0
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        self._sort_column = sort_column
        self._ascending = ascending
        self._title = ""
        self._factor = factor
        self._display_float_auto = []
        self._display_int_auto = []

    @property
    def sort_column(self) -> str:
        """stable_feature sort_column of the report property."""
        return self._sort_column

    @sort_column.setter
    def sort_column(self, value: str) -> None:
        """Setter for the stable_feature sort_column of the report property."""
        self._sort_column = value

    @property
    def ascending(self) -> bool:
        """stable_feature ascending of the report property."""
        return self._ascending

    @ascending.setter
    def ascending(self, value: bool) -> None:
        """Setter for the stable_feature ascending of the report property."""
        self._ascending = value

    @property
    def title(self) -> str:
        """stable_feature title property."""
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        """Setter stable_feature title property."""
        self._title = value

    @property
    def factor(self) -> float:
        """The factor of the table"""
        return self._factor

    @property
    def display_float_auto(self) -> List[str]:
        """display list of columns float values automatically formatted."""
        return self._display_float_auto

    @display_float_auto.setter
    def display_float_auto(self, value: List[str]) -> None:
        """Setter for display list of columns float values automatically formatted."""
        self._display_float_auto = value

    @property
    def display_int_auto(self) -> List[str]:
        """display list of columns integer values automatically formatted."""
        return self._display_int_auto

    @display_int_auto.setter
    def display_int_auto(self, value: List[str]) -> None:
        """Setter for display list of columns integer values automatically formatted."""
        self._display_int_auto = value

    def merge_result(self, result: Result) -> None:
        """
        Merge two TableResults.

        :param: result: The TableResult to merge.
        :return: None
        """
        if not isinstance(result, TableResult):
            raise ValueError(f'result is not of type TableResult')

        # If factor is zero then there is nothing to merge.
        if result.factor == 0:
            return

        # Create new DataFrame of the combined data.
        columns = self._value.columns
        factor = self._factor + result.factor
        merge_df = self._value.merge(result.value, how='outer', on=columns[0])
        merge_df.fillna(value=0, inplace=True)
        for i in range(1, len(columns)):
            self_s = merge_df[f'{columns[i]}_x'] .mul(other=self._factor)
            result_s = merge_df[f'{columns[i]}_y'].mul(other=result.factor)
            merge_s = self_s.add(result_s)
            merge_s = merge_s.div(other=factor)
            merge_df[columns[i]] = merge_s
        self._value = merge_df[columns]
        self._factor = factor
        return

    def sort_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the display columns for the result.

        :param: df:  DataFrame to perform upon.
        :return: DataFrame with the columns for display.
        """
        # Sort if the column is defined.
        if self.sort_column:
            if self.sort_column in df.columns:
                return df.sort_values(by=self.sort_column, axis=0, ascending=self.ascending)
            else:
                logging.warning(f'Sort column {self.sort_column} is not in {self.title} result.')
        return df

    @staticmethod
    def calc_float_format(series: pd.Series) -> str:
        """
        Calculate the display format for float numbers.

        :param: series: Pandas Series of float numbers.
        :return: Display format
        """
        if series.empty:
            return f"6.1"

        min_values = [abs(x) for x in series.unique()]
        min_values.sort()
        if min_values[0] == 0.0 and len(min_values) > 1:
            min_value = min_values[1]
        else:
            min_value = min_values[0]
        if min_value < 0.000001:
            min_size = 6
        else:
            min_size = 1
            while min_value < 1:
                min_size = min_size + 1
                min_value = min_value * 10
        max_size = len(str(int(series.max())))
        return f"{max_size}.{min_size}"

    def create_pretty_table(self, display_df: pd.DataFrame) -> PrettyTable:
        """
        Create the pretty table for the display data.

        :param: display_df:  DataFrame to perform upon.
        :return: A PrettyTable
        """
        count = display_df.shape[0]
        if self.display_number_lines is not None and self.display_number_lines > 0:
            count = min(count, self.display_number_lines)
        pt = PrettyTable()
        pt.title = self.title
        if hasattr(self.value, 'name'):
            pt.title = self.value.name
        pt.field_names = display_df.columns
        for i in range(count):
            cur_row = display_df.iloc[i]
            for col in self._display_int_auto:
                if isinstance(cur_row[col], float) and not np.isnan(cur_row[col]) and \
                        cur_row[col].is_integer():
                    cur_row[col] = int(cur_row[col])
            pt.add_row(row=cur_row)
        pt.align = 'r'
        pt.align[display_df.columns[0]] = 'l'
        pt.float_format = '1.8'

        for col in self._display_float_auto:
            if col in display_df:
                col_format = self.calc_float_format(display_df[col])
                pt.float_format[col] = col_format

        return pt

    def display_value(self) -> str:
        """The display string for the result's value."""
        display_df = self.value.copy()
        display_df = self.sort_display(display_df)

        pt = self.create_pretty_table(display_df)
        output: str = '\n'
        output += str(pt)
        return output

    def save(self, save_dir: str) -> None:
        """
        Save full detail of table result to a file.

        :param: save_dir: string specifying path to folder to save the file
        """
        save_path = Path(save_dir)
        if not save_path.exists() or not save_path.is_dir():
            return
        file_path = save_path.joinpath(f'{self.description}.csv')
        self.value.to_csv(path_or_buf=file_path, index=False)
