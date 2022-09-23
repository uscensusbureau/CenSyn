import logging
import os
import time
from pathlib import Path
from abc import ABC
from typing import Any, Dict

from .report_level import ReportLevel, get_report_level
from censyn.results import (ResultLevel, FloatResult, Result, ParquetResult, ListResult,
                            PlotResult, TableResult, StableFeatureResult)


class Report(ABC):
    """
    A Report takes a List[Result], which are obtained from a given set of Metrics that are run, and outputs the Metrics
    based on the List of Result instances passed in.
    """
    def __init__(self, *args, config: Dict, **kwargs) -> None:
        """
        Constructor for Report.

        :param: config: Configuration containing the parameters for report.
        """
        self._level = ReportLevel.SUMMARY
        self._header = ''
        self._save_path = ""
        self._full_file_path = ""
        self._save_files = True
        self._display_number_lines = 40
        self._stable_feature_display_number_lines = 0
        self._stable_feature_minimum_data_count = 0
        self._stable_feature_display_density = False
        self._stable_feature_sort_column = None
        self._stable_feature_ascending = None
        self._bins_readable_values = True
        self._features = {}
        self._results = {}
        self._append = True

        # Load the configuration data
        self.load_config(config)

    @property
    def header(self) -> str:
        """Header of the report property."""
        return self._header

    @header.setter
    def header(self, value: str) -> None:
        """Setter for the header of the report property."""
        self._header = value

    @property
    def full_file_path(self) -> str:
        """The full file path of report property."""
        return self._full_file_path

    @property
    def save_files(self) -> bool:
        """save_files of the report property."""
        return self._save_files

    @save_files.setter
    def save_files(self, value: bool) -> None:
        """Setter for the save_files of the report property."""
        self._save_files = value

    @property
    def display_number_lines(self) -> int:
        """display_number_lines of the report property."""
        return self._display_number_lines

    @display_number_lines.setter
    def display_number_lines(self, value: int) -> None:
        """Setter for the display_number_lines of the report property."""
        self._display_number_lines = value

    @property
    def stable_feature_display_number_lines(self) -> int:
        """stable_feature_display_number_lines of the report property."""
        return self._stable_feature_display_number_lines

    @stable_feature_display_number_lines.setter
    def stable_feature_display_number_lines(self, value: int) -> None:
        """Setter for the stable_feature_display_number_lines of the report property."""
        self._stable_feature_display_number_lines = value

    @property
    def stable_feature_minimum_data_count(self) -> float:
        """stable_feature_minimum_data_count of the report property."""
        return self._stable_feature_minimum_data_count

    @stable_feature_minimum_data_count.setter
    def stable_feature_minimum_data_count(self, value: float) -> None:
        """Setter for the stable_feature_minimum_data_count of the report property."""
        self._stable_feature_minimum_data_count = value

    @property
    def stable_feature_display_density(self) -> bool:
        """stable_feature_display_density of the report property."""
        return self._stable_feature_display_density

    @stable_feature_display_density.setter
    def stable_feature_display_density(self, value: bool) -> None:
        """Setter for the stable_feature_display_density of the report property."""
        self._stable_feature_display_density = value

    @property
    def stable_feature_sort_column(self) -> str:
        """stable_feature_sort_column of the report property."""
        return self._stable_feature_sort_column

    @stable_feature_sort_column.setter
    def stable_feature_sort_column(self, value: str) -> None:
        """Setter for the stable_feature_sort_column of the report property."""
        self._stable_feature_sort_column = value

    @property
    def stable_feature_ascending(self) -> bool:
        """stable_feature_ascending of the report property."""
        return self._stable_feature_ascending

    @stable_feature_ascending.setter
    def stable_feature_ascending(self, value: bool) -> None:
        """Setter for the stable_feature_ascending of the report property."""
        self._stable_feature_ascending = value

    @property
    def bins_readable_values(self) -> bool:
        """_bins_readable_values of the report property."""
        return self._bins_readable_values

    @bins_readable_values.setter
    def bins_readable_values(self, value: bool) -> None:
        """Setter for the _bins_readable_values of the report property."""
        self._bins_readable_values = value

    @property
    def features(self) -> Dict:
        """features of the report property."""
        return self._features

    @features.setter
    def features(self, value: Dict) -> None:
        """Setter for the features of the report property."""
        self._features = value

    @property
    def level(self) -> ReportLevel:
        """Getter for the report level property."""
        return self._level

    @level.setter
    def level(self, value: ReportLevel) -> None:
        """Setter for the level of the report property."""
        self._level = value

    @property
    def results(self) -> Dict[str, Result]:
        """Getter for results"""
        return self._results

    def load_config(self, config: Dict):
        """
        Load the configuration.

        :param: config: Dictionary of configurations.
        :return: None
        """
        self._level = get_report_level(config.get('report_level',  self._level))
        logging.debug(msg=f"Set report_level to {self._level}.")
        self._bins_readable_values = config.get('bins_readable_values', self._bins_readable_values)
        logging.debug(msg=f"Set bins_readable_values to {self._bins_readable_values}.")
        self._display_number_lines = config.get('display_number_lines', self._display_number_lines)
        self._stable_feature_display_number_lines = config.get('stable_feature_display_number_lines',
                                                               self._stable_feature_display_number_lines)
        logging.debug(msg=f"Set stable_feature_display_number_lines to {self._stable_feature_display_number_lines}.")
        self._stable_feature_minimum_data_count = config.get('stable_feature_minimum_data_count',
                                                             self._stable_feature_minimum_data_count)
        logging.debug(msg=f"Set stable_feature_minimum_data_count to {self._stable_feature_minimum_data_count}.")
        self._stable_feature_display_density = config.get('stable_feature_display_density',
                                                          self._stable_feature_display_density)
        logging.debug(msg=f"Set stable_feature_display_density to {self._stable_feature_display_density}.")
        self._stable_feature_sort_column = config.get('stable_feature_sort_column',
                                                      self._stable_feature_sort_column)
        logging.debug(msg=f"Set stable_feature_sort_column to {self._stable_feature_sort_column}.")
        self._stable_feature_ascending = config.get('stable_feature_ascending',
                                                    self._stable_feature_ascending)
        logging.debug(msg=f"Set stable_feature_ascending to {self._stable_feature_ascending}.")

    def produce_report(self) -> None:
        """
        Process to the formats and displays the evaluation_results to the report.

        :return: None
        """
        if self._results is None:
            raise RuntimeError('No results loaded for Report.')
        elif len(self._results) == 0:
            raise RuntimeError('Empty Dict of results loaded for Report')

        self._append = True
        self._produce_header()

        self._produce_results()

    def _produce_header(self) -> None:
        """Appends the header to the report."""
        self._append_to_report(self.header)

    def _produce_results(self) -> None:
        """Appends the results to the report"""
        for results in self._results.values():
            self._append_result_to_report(results)

    def _append_result_to_report(self, result: Result) -> None:
        """
        Appends the result to the report.

        :param: result: Result to append to the report.
        :return: None
        """
        if result.container:
            for v in iter(result.value):
                if v is not None:
                    self._append_result_to_report(v)
            return

        if isinstance(result, StableFeatureResult):
            result.display_number_lines = self.stable_feature_display_number_lines
            result.display_minimum_data_count = self.stable_feature_minimum_data_count
            result.display_density = self.stable_feature_display_density
            if self.stable_feature_sort_column is not None:
                result.sort_column = self.stable_feature_sort_column
            if self.stable_feature_ascending is not None:
                result.ascending = self.stable_feature_ascending
            if self.bins_readable_values:
                result.display_bins_readable_values(self.features)
        elif result.display_number_lines is None:
            result.display_number_lines = self.display_number_lines

        # Appends the result to the report.
        if self._append:
            if self._valid_display_result(result):
                self._append_to_report(result.display())
                self._append_to_report('')

            self._save_result(result)

    def _save_result(self, result: Result) -> None:
        """
        Saves the result.

        :param: result: Result to save.
        :return: None
        """
        pass

    def _valid_display_result(self, result: Result) -> bool:
        """
        The result is valid for display.

        :param: result: Result to display in the report.
        :return: The display is valid for the report.
        """
        return True

    def _append_to_report(self, text: str) -> None:
        """
        Appends the text to the report.

        :param: text: Text to append.
        :return: None
        """
        pass

    def add_result(self, key: str, value: Result) -> None:
        """
        Add a single result.

        :param: key: keu for the result.
        :param: value: Result to add
        :return: None
        """
        self._results[key] = value

    def time_function(self, key: str, function, kwargs: Dict) -> Any:
        """
        A function to time another function and add it to the current report.

        :param: key {str} -- The key to use in the evaluation result of Report
        :param: function {[type]} -- The function you want to time
        :param: kwargs {Dict} -- Keyword arguments that you want to send the function being run.
        :return: whatever is returned from the passed in function.
        """
        start_time = time.time()
        to_return = function(**kwargs)
        duration = time.time() - start_time
        existing_time = self._results.get(key, None)
        if existing_time:
            duration = duration + existing_time.value
        self.add_result(key, FloatResult(value=duration, metric_name=key))
        return to_return


class ConsoleReport(Report):
    """Report subclass that displays evaluation results to console."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor for ConsoleReport

        :param: args: positional arguments for parent constructor
        :param: kwargs: keyword arguments for parent constructor
        """
        super().__init__(*args, **kwargs)

    def _produce_header(self) -> None:
        """Appends the header to the report."""
        print('ConsoleReport')
        super()._produce_header()

    def _valid_display_result(self, result: Result) -> bool:
        """
        The result is valid for display.

        :param: result: Result to display in the report.
        :return: The display is valid for the report.
        """
        # Check nested results
        if result.container:
            for v in iter(result.value):
                if v is not None:
                    if not self._valid_display_result(v):
                        return False

        # Test levels
        if self.level <= ReportLevel.FULL:
            if result.level > ResultLevel.GENERAL:
                return False
        return True

    def _append_to_report(self, text: str) -> None:
        """
        Appends the text to the report.

        :param: text: Text to append.
        :return: None
        """
        if self._append:
            print(text)


class FileReport(Report):
    """Report subclass that displays evaluation results to file."""

    def __init__(self, file_full_path: str, rename_if_exists: bool = False, sub_path: str = "",
                 *args, **kwargs) -> None:
        """
        Constructor for FileReport.

        :param: file_full_path: full file path for FileReport
        :param: rename_if_exists: Boolean flag to rename output report file if currently exists.
        :param: sub_path: Path to append for the output file reports.
        :param: args: positional arguments for parent constructor
        :param: kwargs: keyword arguments for parent constructor
        """
        super().__init__(*args, **kwargs)

        file_path, self._file_name = os.path.split(file_full_path)
        if sub_path:
            file_path = os.path.join(file_path, sub_path)
        root, ext = os.path.splitext(self._file_name)
        real_file_path = os.path.join(file_path, f'{root}{ext}')
        self._save_path = os.path.join(file_path, root)
        if not os.path.exists(file_path):
            msg = f"Report path {file_path} does not exist."
            logging.error(msg=msg)
            raise ValueError(msg)
        if os.path.exists(real_file_path):
            attempt = 0
            while os.path.exists(real_file_path):
                if rename_if_exists:
                    if attempt > 100:
                        msg = f"You have attempted to write to this file too many times. Please choose a new file name."
                        logging.error(msg=msg)
                        raise ValueError(msg)
                    attempt += 1
                    real_file_path = os.path.join(file_path, f'{root}_{str(attempt)}{ext}')
                else:
                    raise ValueError('You provided a file name to FileReport that already exists.')
            self._save_path = os.path.join(file_path, f'{root}_{str(attempt)}')
        Path(self._save_path).mkdir(exist_ok=True)
        self._full_file_path = real_file_path
        self._open_file = None

    def load_config(self, config: Dict) -> None:
        """
        Load the configuration.

        :param: config: Dictionary of configurations.
        :return: None
        """
        super().load_config(config)

    def produce_report(self) -> None:
        """Process to the formats and displays the evaluation_results to the report."""
        Path(self._save_path).mkdir(exist_ok=True)

        with open(self.full_file_path, 'w') as self._open_file:
            super().produce_report()

    def _valid_display_result(self, result: Result) -> bool:
        """
        The result is valid for display.

        :param: result: Result to display in the report.
        :return: The display is valid for the report.
        """
        # Check nested results
        if result.container:
            for v in iter(result.value):
                if v is not None:
                    if not self._valid_display_result(v):
                        return False

        # Test levels
        if self.level <= ReportLevel.SUMMARY:
            if result.level > ResultLevel.GENERAL:
                return False

        return True

    def _append_to_report(self, text: str) -> None:
        """Appends the text to the report."""
        if self._append:
            self._open_file.write(f'{text}\n')

    def _save_result(self, result: Result) -> None:
        """
        Checks the save_files before saving the result.

        :param: result: Result to save.
        :return: None
        """
        if self._save_files:
            if isinstance(result, (ParquetResult, PlotResult)):
                if self.level >= ReportLevel.FULL:
                    result.save(self._save_path)
            elif isinstance(result, ListResult):
                result.save(self._save_path)
            elif isinstance(result, TableResult):
                result.save(self._save_path)
            elif isinstance(result, StableFeatureResult):
                result.save(self._save_path)
