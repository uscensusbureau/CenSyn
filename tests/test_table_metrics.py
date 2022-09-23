import unittest

import pandas as pd

from censyn.metrics import TableMetrics
from censyn.results import result


class TableMetricTest(unittest.TestCase):

    def test_groupby_results(self) -> None:
        """
        There should be no differences between the two simulated datasets
        """

        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        table_metric = TableMetrics(features=['one', 'two'], name='TEST')

        # (Result, list) containing all the component Result objs
        tab_met_res = table_metric.compute_results(data_frame_a, data_frame_b, None, None, None, None)
        self.assertIsInstance(tab_met_res, result.PandasResult)

        component_results = tab_met_res.value
        self.assertEqual(component_results['error'].sum(), 0)

    def test_pivot_results(self) -> None:
        """
        There should be no differences between the two simulated datasets
        """

        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        table_metric = TableMetrics(features=['one', 'two'], name='TEST', pivot=True)

        # (Result, list) containing all the component Result objs
        tab_met_res = table_metric.compute_results(data_frame_a, data_frame_b, None, None, None, None)
        self.assertIsInstance(tab_met_res, result.PandasResult)

        component_results = tab_met_res.value
        self.assertEqual(component_results.loc[3, 33], 0)


if __name__ == '__main__':
    unittest.main()
