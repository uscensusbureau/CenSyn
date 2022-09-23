import unittest

import pandas as pd

from censyn.results import ResultLevel
from censyn.results import FloatResult, IndexResult, IntResult, ListResult, MappingResult, StrResult, \
    StableFeatureResult
from censyn.utils import StableFeatures


class TestResult(unittest.TestCase):

    def test_IntResult(self) -> None:
        int_result = IntResult(value=3, metric_name='dummy_metric')

        # test value
        self.assertEqual(int_result.value, 3)

        # test metric_name
        self.assertEqual(int_result.metric_name, 'dummy_metric')

    def test_FloatResult(self) -> None:
        float_result = FloatResult(value=12.5, metric_name='dummy_metric')

        # test value
        self.assertEqual(float_result.value, 12.5)

        # test metric_name
        self.assertEqual(float_result.metric_name, 'dummy_metric')

    def test_StrResult(self) -> None:
        str_result = StrResult(value='str_value', metric_name='dummy_metric')

        # test value
        self.assertEqual(str_result.value, 'str_value')

        # test metric_name
        self.assertEqual(str_result.metric_name, 'dummy_metric')

    def test_IndexResult(self) -> None:
        list_result = IndexResult(value=[1, 2, 3], metric_name='dummy_metric')

        # test value
        self.assertEqual(list_result.value, [1, 2, 3])

        # test metric_name
        self.assertEqual(list_result.metric_name, 'dummy_metric')

        # test initialization on empty list
        list_result = ListResult(value=[], metric_name='dummy_metric')
        self.assertEqual(list_result.value, [])

    def test_ListResult(self) -> None:
        list_result = ListResult(value=[1, 2, 3], metric_name='dummy_metric')

        # test value
        self.assertEqual(list_result.value, [1, 2, 3])

        # test metric_name
        self.assertEqual(list_result.metric_name, 'dummy_metric')

        # test initialization on empty list
        list_result = ListResult(value=[], metric_name='dummy_metric')
        self.assertEqual(list_result.value, [])

    def test_MappingResult(self) -> None:
        mapping_result = MappingResult(value={'dog': 1, 'cat': 2}, metric_name='dummy_metric')

        # test value
        self.assertEqual(mapping_result.value, {'dog': 1, 'cat': 2})

        # test metric_name
        self.assertEqual(mapping_result.metric_name, 'dummy_metric')

        # test initialization on empty list
        list_result = MappingResult(value={}, metric_name='dummy_metric')
        self.assertEqual(list_result.value, {})

    def test_StableFeatureResult(self) -> None:
        """Test for StableFeatures."""
        data_a = [0, 1, 0, 1, 2, 1, 0, 2, 1, 0, 1, 1]
        data_b = [0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1]
        density_a = [data_a[i] / sum(data_a) for i in range(len(data_a))]
        density_b = [data_b[i] / sum(data_b) for i in range(len(data_b))]
        diff = [abs(density_a[i] - density_b[i]) for i in range(len(data_a))]
        arrays = [['a', 'b', 'c'], [1, 2], ['Y', 'Z']]

        index = pd.MultiIndex.from_product(arrays, names=['first', 'second', 'third'])
        test_df = pd.DataFrame({0: diff, 1: density_a, 2: density_b}, index=index)
        features = ['first']
        sf = StableFeatures(names=features)
        sf.add_scores(test_df)
        sf_result = StableFeatureResult(value=sf.scores(), metric_name='test_StableFeatureResult',
                                        level=ResultLevel.SUMMARY,
                                        description=f"Density Distribution for {features}",
                                        sf=features,
                                        data_a_count=sum(data_a),
                                        data_b_count=sum(data_b),
                                        baseline_count=0)
        out = sf_result.display()
        out_lines = out.splitlines()
        self.assertEqual(out_lines[6].split(sep='|')[1].strip(), "c")
        self.assertEqual(out_lines[7].split(sep='|')[1].strip(), "a")
        self.assertEqual(out_lines[8].split(sep='|')[1].strip(), "b")
        self.assertEqual(7, len(out_lines[8].split(sep='|')))
        sf_result.display_density = True
        sf_result.sort_column = 'Bins'
        sf_result.ascending = False
        out = sf_result.display()
        out_lines = out.splitlines()
        self.assertEqual(out_lines[6].split(sep='|')[1].strip(), "c")
        self.assertEqual(out_lines[7].split(sep='|')[1].strip(), "b")
        self.assertEqual(out_lines[8].split(sep='|')[1].strip(), "a")
        self.assertEqual(9, len(out_lines[8].split(sep='|')))


if __name__ == '__main__':
    unittest.main()
