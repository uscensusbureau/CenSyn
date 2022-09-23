import io
import os
import shutil
import sys
import unittest
from pathlib import Path

from censyn.metrics import PickingStrategy
from censyn.report import report
from censyn.results import result


class TestReport(unittest.TestCase):
    """
    Each Report object takes a List[Result], which are obtained from a given set of Metrics that are run,
    and displays based on the List of Result instances passed in.
    """

    _file_dir = Path(__file__).resolve().parent.parent / 'tests' / 'assets'
    _file_name = 'file_report'
    _file_path = str(_file_dir / (_file_name + '.txt'))

    def setUp(self) -> None:
        self.evaluation_output = 'Metric: int_metric, int_result: 3\n\n' \
                          'Metric: float_metric, float_result: 12.5\n\n' \
                          'Metric: str_metric, str_result: test_string\n\n' \
                          'Metric: list_metric, list_result: \n' \
                          '1\n' \
                          '2\n' \
                          '3\n\n' \
                          'Metric: mapping_metric, mapping_result: \n' \
                          'dog: 1\n' \
                          'cat: 2\n\n'

        int_result = result.IntResult(value=3, metric_name='int_metric', description='int_result')
        float_result = result.FloatResult(value=12.5, metric_name='float_metric', description='float_result')
        str_result = result.StrResult(value='test_string', metric_name='str_metric', description='str_result')
        list_result = result.ListResult(value=[1, 2, 3], metric_name='list_metric', description='list_result')
        mapping_result = result.MappingResult(value={'dog': 1, 'cat': 2}, metric_name='mapping_metric',
                                              description='mapping_result')

        self.evaluation_results = {
            'int_metric':     int_result,
            'float_metric':   float_result,
            'str_metric':     str_result,
            'list_metric':    list_result,
            'mapping_metric': mapping_result
        }

        if os.path.exists(self._file_path):
            os.remove(self._file_path)

    def tearDown(self) -> None:
        """Cleans up the file if it wasn't properly removed already."""
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        isdir = os.path.isdir(self._file_dir / self._file_name)
        if isdir:
            shutil.rmtree(self._file_dir / self._file_name, ignore_errors=True)

    def test_console_report(self) -> None:
        # setup the environment
        backup = sys.stdout
        sys.stdout = io.StringIO()  # capture output

        report_cfg = {
            "report_level": "FULL"
        }
        console_report = report.ConsoleReport(config=report_cfg)
        for k, v in self.evaluation_results.items():
            console_report.add_result(key=k, value=v)
        console_report.produce_report()

        output = sys.stdout.getvalue()
        sys.stdout = backup  # restore original stdout

        expected_output = f'ConsoleReport\n\n{self.evaluation_output}'
        self.assertEqual(output, expected_output)  # test that obtained expected output

    def test_file_report(self) -> None:
        self.assertEqual.__self__.maxDiff = None
        report_cfg = {
            "report_level": "FULL"
        }
        file_report = report.FileReport(file_full_path=self._file_path, config=report_cfg)
        for k, v in self.evaluation_results.items():
            file_report.add_result(key=k, value=v)

        report_header = 'Summary: \n' \
                        'This report was generated using the 3 Marginal Metric approach\n' \
                        f'- picking strategy: {PickingStrategy.rolling.name}\n' \
                        f'- sample ratio: {1.0}\n' \
                        'How to interpret results - the marginal metric score (value between 0 and 2):\n' \
                        '  0 - perfectly matching density distributions (for the marginals used in the comparison).\n' \
                        '  2 - no overlap whatsoever (for the marginals used in the comparison).\n'

        file_report.header = report_header
        file_report.produce_report()

        # read in and validate content
        with open(self._file_path) as file:
            data = file.read()

            expected_output = f'{report_header}\n{self.evaluation_output}'
            self.assertEqual(data, expected_output)  # test that obtained expected output

        self.assertTrue(os.path.isdir(self._file_dir / self._file_name),
                        msg=f"Report folder {self._file_dir / self._file_name} do not exist")

        # try on existing file, except error
        with self.assertRaises(ValueError) as cm:
            file_report = report.FileReport(file_full_path=self._file_path, config=report_cfg)
            for k, v in self.evaluation_results.items():
                file_report.add_result(key=k, value=v)
        err = cm.exception
        self.assertEqual(str(err), 'You provided a file name to FileReport that already exists.')


if __name__ == '__main__':
    unittest.main()
