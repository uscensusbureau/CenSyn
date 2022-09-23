import unittest

import numpy as np
import pandas as pd

from censyn.metrics import CellDifferenceMetric, MarginalMetric, PickingStrategy, DensityDistributionMetric
from censyn.metrics import JoinMarginalMetric
import censyn.metrics as metrics
from censyn.results import result, TableResult
from censyn.utils import (compute_mapping_feature_combinations_to_density_distribution_differences,
                          compute_density_distribution_differences, compute_density_mean)

a_df = pd.DataFrame({'one': ['x', 'x', 'y', 'y', 'x', 'y'],
                     'two': [2, None, 4, 8, 2, 2],
                     'three': [1, 2, 0, 3, 3, 3],
                     'four': [12, 14, 24, None, 10, 14]
                     })
b_df = pd.DataFrame({'one': ['x', 'x', 'x', 'y', 'x', 'y'],
                     'two': [2, 3, 4, 8, 4, 2],
                     'three': [None, 2, 0, 3, 3, 2],
                     'four': [12, 14, 18, None, 10, 14]
                     })
weight_a = pd.Series([10, 10, 20, 10, 40, 10])
weight_b = pd.Series([20, 10, 20, 10, 20, 20])
add_a_df = pd.DataFrame({'serial': [1, 1, 4, 5, 6, 7],
                         'num': [1, 2, 1, 1, 1, 1]})
add_b_df = pd.DataFrame({'serial': [1, 1, 3, 5, 7, 8],
                         'num': [1, 2, 1, 1, 1, 1]})


class CellDifferenceMetricTest(unittest.TestCase):
    def test_difference(self) -> None:
        metric = CellDifferenceMetric(name='test')
        cell_results = metric.compute_results(data_frame_a=a_df, data_frame_b=b_df, weight_a=None, weight_b=None,
                                              add_a=None, add_b=None)
        self.assertIsInstance(cell_results, result.ListResult)
        component_results = cell_results.value
        self.assertEqual(3, len(component_results))
        self.assertIsInstance(component_results[0], result.FloatResult)
        self.assertEqual("Time", component_results[0].description)
        self.assertIsInstance(component_results[1], result.MappingResult)
        self.assertEqual("Summary total", component_results[1].description)
        self.assertEqual(component_results[1].value['Total Differences'], 6)
        self.assertAlmostEqual(component_results[1].value['Total Differences %'], 25.000000, 3)
        self.assertAlmostEqual(component_results[1].value['Average number of differences per row'], 1.000000, 3)
        self.assertIsInstance(component_results[2], TableResult)
        self.assertEqual("Individual feature statistics", component_results[2].description)
        self.assertEqual(6, component_results[2].value['Count'].sum())
        self.assertEqual(2, component_results[2].value['NA Count'].sum())
        self.assertEqual(1, component_results[2].value['Median Difference'].isna().sum())

    def test_difference_error(self) -> None:
        with self.assertRaises(ValueError):
            metric = CellDifferenceMetric(name='test')
            mod_b_df = b_df.drop(index=2, axis=1, inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=mod_b_df,
                                   weight_a=None, weight_b=None, add_a=None, add_b=None)

    def test_additional_error(self) -> None:
        with self.assertRaises(ValueError):
            metric = CellDifferenceMetric(name='test')
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=None, weight_b=None, add_a=add_a_df, add_b=None)

        with self.assertRaises(ValueError):
            metric = CellDifferenceMetric(name='test')
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=None, weight_b=None, add_a=None, add_b=add_b_df)

        with self.assertRaises(ValueError):
            metric = CellDifferenceMetric(name='test')
            mod_b_df = add_b_df.drop(index=4, inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=None, weight_b=None, add_a=add_a_df, add_b=mod_b_df)

        with self.assertRaises(ValueError):
            metric = CellDifferenceMetric(name='test')
            mod_a_df = add_a_df.drop(index=4, inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=None, weight_b=None, add_a=mod_a_df, add_b=add_b_df)

    def test_additional(self) -> None:
        metric = CellDifferenceMetric(name='test')
        cell_results = metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                              weight_a=weight_a, weight_b=weight_b, add_a=add_a_df, add_b=add_b_df)
        self.assertIsInstance(cell_results, result.ListResult)
        component_results = cell_results.value
        self.assertEqual(3, len(component_results))
        self.assertIsInstance(component_results[0], result.FloatResult)
        self.assertEqual("Time", component_results[0].description)
        self.assertIsInstance(component_results[1], result.MappingResult)
        self.assertEqual("Summary total", component_results[1].description)
        self.assertEqual(component_results[1].value['Total Differences'], 5)
        self.assertAlmostEqual(component_results[1].value['Total Differences %'], 31.250000, 3)
        self.assertAlmostEqual(component_results[1].value['Average number of differences per row'], 1.250000, 3)
        self.assertAlmostEqual(component_results[1].value['Weighted Total Differences %'], 35.000000, 3)
        self.assertIsInstance(component_results[2], TableResult)
        self.assertEqual("Individual feature statistics", component_results[2].description)
        self.assertEqual(5, component_results[2].value['Count'].sum())
        self.assertEqual(2, component_results[2].value['NA Count'].sum())
        self.assertEqual(1, component_results[2].value['Median Difference'].isna().sum())


class MarginalMetricTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_data_error(self) -> None:
        with self.assertRaises(ValueError):
            metric = MarginalMetric(name='test')
            mod_b_df = b_df.drop(axis=0, columns=['four'], inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=mod_b_df,
                                   weight_a=None, weight_b=None, add_a=None, add_b=None)

    def test_weight_error(self) -> None:
        with self.assertRaises(ValueError):
            metric = MarginalMetric(name='test')
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=weight_a, weight_b=None, add_a=None, add_b=None)

        with self.assertRaises(ValueError):
            metric = MarginalMetric(name='test')
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=None, weight_b=weight_b, add_a=None, add_b=None)

        with self.assertRaises(ValueError):
            metric = MarginalMetric(name='test')
            mod_weight_b = weight_b.drop(index=4, inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=weight_a, weight_b=mod_weight_b, add_a=None, add_b=None)

        with self.assertRaises(ValueError):
            metric = MarginalMetric(name='test')
            mod_weight_a = weight_b.drop(index=4, inplace=False)
            metric.compute_results(data_frame_a=a_df, data_frame_b=b_df,
                                   weight_a=mod_weight_a, weight_b=weight_b, add_a=None, add_b=None)

    def test_lexicographic_feature_combinations(self) -> None:

        # create data frame with a number of columns
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)
        marginal_metric = metrics.MarginalMetric(marginal_dimensionality=3,
                                                 picking_strategy=PickingStrategy.lexicographic,
                                                 sample_ratio=1.0, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        # check that all feature combinations returned for the specified marginal dimensionality,
        # in lexicographic order; marginal dim = 3
        lexicographic_feature_combinations = marginal_metric.get_feature_combinations()

        self.assertEqual(lexicographic_feature_combinations, [('one', 'two', 'three'),
                                                              ('one', 'two', 'four'),
                                                              ('one', 'two', 'five'),
                                                              ('one', 'three', 'four'),
                                                              ('one', 'three', 'five'),
                                                              ('one', 'four', 'five'),
                                                              ('two', 'three', 'four'),
                                                              ('two', 'three', 'five'),
                                                              ('two', 'four', 'five'),
                                                              ('three', 'four', 'five')])

        # check that first 50% feature combinations returned for marginal dimensionality of 3, in lexicographic order
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.lexicographic, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        lexicographic_feature_combinations = marginal_metric.get_feature_combinations()

        self.assertEqual(lexicographic_feature_combinations, [('one', 'two', 'three'),
                                                              ('one', 'two', 'four'),
                                                              ('one', 'two', 'five'),
                                                              ('one', 'three', 'four'),
                                                              ('one', 'three', 'five')])

        # check that all feature combinations returned for the specified marginal dimensionality,
        # in lexicographic order; marginal dim = 2
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.lexicographic, sample_ratio=1, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        lexicographic_feature_combinations = marginal_metric.get_feature_combinations()

        self.assertEqual(lexicographic_feature_combinations, [('one', 'two'),
                                                              ('one', 'three'),
                                                              ('one', 'four'),
                                                              ('one', 'five'),
                                                              ('two', 'three'),
                                                              ('two', 'four'),
                                                              ('two', 'five'),
                                                              ('three', 'four'),
                                                              ('three', 'five'),
                                                              ('four', 'five')])

        # check that first 20% feature combinations returned for marginal dimensionality of 2, in lexicographic order
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.lexicographic, sample_ratio=0.20, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        lexicographic_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(lexicographic_feature_combinations, [('one', 'two'), ('one', 'three')])

    def test_random_feature_combinations(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.random, sample_ratio=1.0, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        all_valid_combinations = [('one', 'two', 'three'),
                                  ('one', 'two', 'four'),
                                  ('one', 'two', 'five'),
                                  ('one', 'three', 'four'),
                                  ('one', 'three', 'five'),
                                  ('one', 'four', 'five'),
                                  ('two', 'three', 'four'),
                                  ('two', 'three', 'five'),
                                  ('two', 'four', 'five'),
                                  ('three', 'four', 'five')]

        # test that that all are returned if sample_ratio == 1.0
        random_feature_combinations = marginal_metric.get_feature_combinations()

        # test proper number of combinations returned
        self.assertEqual(len(random_feature_combinations), 10)

        # test that combinations returned are in set of valid possible combinations
        self.assertTrue(set(all_valid_combinations).issuperset(set(random_feature_combinations)))

        # test that 50% returned
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.random, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        random_feature_combinations = marginal_metric.get_feature_combinations()

        # test proper number of combinations returned
        self.assertEqual(len(random_feature_combinations), 5)

        # test that combinations returned are in set of valid possible combinations.
        # test this way because result from random_feature_combinations() is non deterministic.
        self.assertTrue(set(all_valid_combinations).issuperset(set(random_feature_combinations)))

    def test_rolling_feature_combinations(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        # check returned features, sample_ratio = 1.0, marginal_dimensionality = 2
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=1.0, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        rolling_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(rolling_feature_combinations, [('one', 'two'), ('two', 'three'),
                                                        ('three', 'four'), ('four', 'five')])

        # check returned features, sample_ratio = 0.5, marginal_dimensionality = 2
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        rolling_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(rolling_feature_combinations, [('one', 'two'), ('two', 'three')])

        # check returned features, sample_ratio = 1.0, marginal_dimensionality = 3
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=1.0, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        rolling_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(rolling_feature_combinations, [('one', 'two', 'three'), ('two', 'three', 'four'),
                                                        ('three', 'four', 'five')])

        # check returned features, sample_ratio = 0.33, marginal_dimensionality = 3
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.33, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        rolling_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(rolling_feature_combinations, [('one', 'two', 'three')])

        # check returned features, sample_ratio = 0.66, marginal_dimensionality = 3
        marginal_metric = MarginalMetric(marginal_dimensionality=3,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.66, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b
        rolling_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(rolling_feature_combinations, [('one', 'two', 'three'), ('two', 'three', 'four')])

    def test_get_feature_combinations(self) -> None:
        """Test that correct combination method is called - these combination methods are already tested above"""

        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.lexicographic, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        lexicographic_feature_combinations = marginal_metric.get_feature_combinations()

        self.assertEqual(lexicographic_feature_combinations, [('one', 'two'),
                                                              ('one', 'three'),
                                                              ('one', 'four'),
                                                              ('one', 'five'),
                                                              ('two', 'three')])

        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.random, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        random_feature_combinations = marginal_metric.get_feature_combinations()

        all_valid_combinations = [('one', 'two'),
                                  ('one', 'three'),
                                  ('one', 'four'),
                                  ('one', 'five'),
                                  ('two', 'three'),
                                  ('two', 'four'),
                                  ('two', 'five'),
                                  ('three', 'four'),
                                  ('three', 'five'),
                                  ('four', 'five')]

        # test proper number of combinations returned; 5 if sample_ratio = 0.5
        self.assertEqual(len(random_feature_combinations), 5)
        # test that combinations returned are in set of valid possible combinations
        self.assertTrue(set(all_valid_combinations).issuperset(set(random_feature_combinations)))

        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        rolling_feature_combinations = marginal_metric.get_feature_combinations()

        self.assertEqual(rolling_feature_combinations, [('one', 'two'), ('two', 'three')])

        column_names = [f"feat{i}" for i in range(10000)]
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.random, sample_ratio=0.001, name='TEST')
        marginal_metric.data_frame_a = pd.DataFrame(columns=column_names)
        marginal_metric.data_frame_b = pd.DataFrame(columns=column_names)

        random_feature_combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(len(random_feature_combinations), 49995)

    def test_compute_mapping_feature_combinations_to_predicates(self) -> None:
        # set up two data frames
        # 5 features: 'one', 'two', 'three', 'four', 'five'
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        # create a marginal metric with marginal dimensionality of 2, rolling picking strategy, and sample ratio of 0.5
        # don't need to test on other parameters for marginal metric construction here,
        # as that has been handled by preceding unit tests.
        # marginal metric with rolling picking strategy and sample_ratio so that only two 2-marginals are computed.
        # keeps testing simpler
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        # test proper feature combinations
        feature_combinations_to_predicates_a, feature_combinations_to_predicates_b = \
            marginal_metric.compute_mapping_feature_combinations_to_predicates()

        feature_combos_a = list(feature_combinations_to_predicates_a.keys())
        feature_combos_b = list(feature_combinations_to_predicates_b.keys())

        self.assertEqual(feature_combos_a, ["('one', 'two')", "('two', 'three')"])
        self.assertEqual(feature_combos_b, ["('one', 'two')", "('two', 'three')"])

        # test proper predicates for each feature combination
        # observed predicates should consist of two lists: the first is the per-row values for feature 'one' and 'two'
        # the second list is the per-row values for feature 'two' and 'three'
        observed_predicates_a = list(feature_combinations_to_predicates_a.values())
        self.assertEqual(observed_predicates_a, [[(1, 11), (2, 22), (3, 33)], [(11, 111), (22, 222), (33, 333)]])

        observed_predicates_b = list(feature_combinations_to_predicates_b.values())
        self.assertEqual(observed_predicates_b, [[(3, 33), (4, 44), (5, 55)], [(33, 333), (44, 444), (55, 555)]])

    def test_compute_mapping_feature_combinations_to_density_distributions(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        # create a marginal metric with marginal dimensionality of 2, rolling picking strategy, and sample ratio of 0.5
        # don't need to test on other parameters for marginal metric construction here,
        # as that has been handled by preceding unit tests.
        # marginal metric with rolling picking strategy and sample_ratio so that only two 2-marginals are computed.
        # keeps testing simpler
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        feature_combinations = marginal_metric.get_feature_combinations()
        feature_map_a = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_a, None)
        feature_map_b = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_b, None)
        density_dist_diffs = compute_mapping_feature_combinations_to_density_distribution_differences(
            feature_map_a, feature_map_b)

        # check feature combos for both data frames
        feature_combos_a = list(density_dist_diffs.keys())
        feature_combos_b = list(density_dist_diffs.keys())
        self.assertEqual(feature_combos_a.sort(), [('one', 'two'), ('two', 'three')].sort())
        self.assertEqual(feature_combos_b.sort(), [('one', 'two'), ('two', 'three')].sort())

        # check density_distributions
        density_distributions_a = list(density_dist_diffs.values())
        density_distributions_b = list(density_dist_diffs.values())

        expected_dense_dist_a = np.array([0.33333333, 0.33333333, 0.3333333, 0.0, 0.0])
        expected_dense_dist_b = np.array([0.0, 0.0, 0.33333333, 0.33333333, 0.33333333])

        # check density distributions for the two feature combos, for both data frame
        self.assertTrue(np.allclose(density_distributions_a[0]['0_A'].values, expected_dense_dist_a))
        self.assertTrue(np.allclose(density_distributions_b[0]['0_B'].values, expected_dense_dist_b))
        self.assertTrue(np.allclose(density_distributions_a[1]['0_A'].values, expected_dense_dist_a))
        self.assertTrue(np.allclose(density_distributions_b[1]['0_B'].values, expected_dense_dist_b))

        # test tolerance values are small enough
        self.assertFalse(np.allclose(density_distributions_b[1]['0_B'].values, expected_dense_dist_a))

    def test_compute_mapping_feature_combinations_to_density_distribution_differences(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        # create a marginal metric with marginal dimensionality of 2, rolling picking strategy, and sample ratio of 0.5
        # don't need to test on other parameters for marginal metric construction here,
        # as that has been handled by preceding unit tests
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        feature_combinations = marginal_metric.get_feature_combinations()
        feature_map_a = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_a, None)
        feature_map_b = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_b, None)
        feature_combinations_to_absolute_density_difference = \
            compute_mapping_feature_combinations_to_density_distribution_differences(feature_map_a, feature_map_b)

        # test density_distribution_abs_diff are correct
        expected_dense_diff = np.array([[0.33333333],
                                        [0.33333333],
                                        [0.0],
                                        [0.3333333],
                                        [0.3333333]])

        first_dense_diff = feature_combinations_to_absolute_density_difference[('one', 'two')]
        second_dense_diff = feature_combinations_to_absolute_density_difference[('two', 'three')]

        for i in range(len(expected_dense_diff)):
            expect = expected_dense_diff[i][0]
            self.assertAlmostEqual(expect, first_dense_diff['diff'].values[i], places=5)
            self.assertAlmostEqual(expect, second_dense_diff['diff'].values[i], places=5)

    def test_compute_mean_density_distribution_differences(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        # create a marginal metric with marginal dimensionality of 2, rolling picking strategy, and sample ratio of 0.5
        # don't need to test on other parameters for marginal metric construction here,
        # as that has been handled by preceding unit tests
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5, name='TEST')
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        feature_combinations = marginal_metric.get_feature_combinations()
        feature_map_a = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_a, None)
        feature_map_b = marginal_metric.compute_combinations_density_distributions(feature_combinations,
                                                                                   data_frame_b, None)
        density_dist_diffs = compute_mapping_feature_combinations_to_density_distribution_differences(
            feature_map_a, feature_map_b)
        calculated_density_dist_diffs = compute_density_distribution_differences(density_dist_diffs)
        mean_density_dist_diffs = compute_density_mean(calculated_density_dist_diffs)

        # Check correct mean computed.
        # Because the summed abs difference between density distribution for both 2-marginals is 1.33333333,
        # the mean should be the same value.
        # The summed abs difference between the two data sets for both density distributions
        # (2-marginal ('one', 'two') and 2-marginal ('two', 'three') is 1.3333333
        # because the abs difference between the two data sets for both density distributions is
        # [0.33333333], [0.33333333], [0.0], [0.3333333], [0.3333333]]
        self.assertAlmostEqual(mean_density_dist_diffs, 1.3333333)

    def test_compute_results(self) -> None:
        """The values being returned in result objects here have already been covered by previous test cases -
         only testing that expected Results are being returned"""
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])

        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])

        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        data_frame_frequent_item_set = pd.DataFrame(columns=['itemsets', 'count', 'sum score', 'average score'])
        data_frame_frequent_item_set['itemsets'] = pd.Series([["three"], ["two"], ["three", "two"]])
        data_frame_frequent_item_set['count'] = pd.Series([1, 1, 1], dtype=int)
        data_frame_frequent_item_set['sum score'] = pd.Series([1.33333, 1.33333, 1.333333])
        data_frame_frequent_item_set['average score'] = pd.Series([1.33333, 1.33333, 1.333333])

        # create a marginal metric with marginal dimensionality of 2, rolling picking strategy, and sample ratio of 0.5
        # don't need to test on other parameters for marginal metric construction here, as that has been handled by
        # preceding unit tests marginal metric with rolling picking strategy and sample_ratio so that only two
        # 2-marginals are computed - keeps testing simpler
        marginal_metric = MarginalMetric(marginal_dimensionality=2,
                                         picking_strategy=PickingStrategy.rolling, sample_ratio=0.5,
                                         stable_features=None, name='TEST')
        # (Result, list) containing all the component Result objects
        mar_met_res = marginal_metric.compute_results(data_frame_a, data_frame_b, None, None, None, None)

        self.assertIsInstance(mar_met_res, result.ListResult)

        component_results = mar_met_res.value

        stable_features_result = component_results[1]
        self.assertIsNone(stable_features_result)

        # (Result, dict) - mapping of feature combos to density distributions for data set a
        dens_dist_a_result = component_results[3]
        self.assertIsInstance(dens_dist_a_result, result.MappingResult)
        # (Result, dict) - mapping of feature combos to density distributions for data set b
        dens_dist_b_result = component_results[4]
        self.assertIsInstance(dens_dist_b_result, result.MappingResult)
        # (Result, float) Mean( Sum( Abs_Dif( Density Distributions) ) ), of both data sets
        marginal_score_result = component_results[0]
        self.assertIsInstance(marginal_score_result, result.MappingResult)

        frequent_item_set_result = component_results[2]
        self.assertIsInstance(frequent_item_set_result, TableResult)
        frequent_item_set = frequent_item_set_result.value
        pd.testing.assert_frame_equal(frequent_item_set, data_frame_frequent_item_set)

    def test_stable_combinations(self) -> None:
        column_names = ['one', 'two', 'three', 'four', 'five']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)
        marginal_metric = MarginalMetric(marginal_dimensionality=2, name='TEST',
                                         picking_strategy=PickingStrategy.lexicographic,
                                         sample_ratio=1.0, stable_features=['one'])
        marginal_metric.data_frame_a = data_frame_a
        marginal_metric.data_frame_b = data_frame_b

        combinations = marginal_metric.get_feature_combinations()
        self.assertEqual(len(combinations), 6)

    def test_density_distribution(self) -> None:
        column_names = ['one', 'two', 'three']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['one'] = pd.Series([1, 2, 3])
        data_frame_b['one'] = pd.Series([3, 4, 5])
        data_frame_a['two'] = pd.Series([11, 22, 33])
        data_frame_b['two'] = pd.Series([33, 44, 55])
        data_frame_a['three'] = pd.Series([111, 222, 333])
        data_frame_b['three'] = pd.Series([333, 444, 555])

        dd_metric = DensityDistributionMetric(name='TEST')
        dd_results = dd_metric.compute_results(data_frame_a, data_frame_b, None, None, None, None)
        self.assertIsInstance(dd_results, result.ListResult)
        component_results = dd_results.value
        self.assertIsInstance(component_results[0], result.MappingResult)
        self.assertEqual("summary of results", component_results[0].description)
        self.assertIsInstance(component_results[1], result.ListResult)
        self.assertEqual("Feature error scores", component_results[1].description)
        self.assertIsInstance(component_results[2], result.ListResult)

    def test_join_marginal_metric(self) -> None:
        column_names = ['serial_no', 'one', 'two', 'three']
        data_frame_a = pd.DataFrame(columns=column_names)
        data_frame_b = pd.DataFrame(columns=column_names)

        data_frame_a['serial_no'] = pd.Series([1, 1, 1, 2, 2, 3, 3, 4, 5, 5])
        data_frame_a['one'] = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        data_frame_a['two'] = pd.Series([21, 22, 23, 24, 21, 22, 23, 24, 21, 22])
        data_frame_a['three'] = pd.Series([31, 32, 33, 34, 31, 32, 33, 34, 35, 35])

        data_frame_b['serial_no'] = pd.Series([1, 1, 1, 2, 2, 3, 3, 4, 5, 5])
        data_frame_b['one'] = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        data_frame_b['two'] = pd.Series([24, 21, 23, 22, 23, 21, 22, 24, 23, 21])
        data_frame_b['three'] = pd.Series([33, 33, 34, 31, 32, 33, 34, 35, 35, 34])

        join_metric = JoinMarginalMetric(name='TEST', join_features=['serial_no'], edge_features=['two'],
                                         duplicate_features=['one'], retain_duplicates=True,
                                         marginal_dimensionality=1, picking_strategy=PickingStrategy.lexicographic,
                                         sample_ratio=1.0, use_bins=False, use_weights=False)
        process_features = ['one', 'two', 'three']
        add_features = join_metric.additional_features
        join_results = join_metric.compute_results(data_frame_a[process_features], data_frame_b[process_features], None,
                                                   None, data_frame_a[add_features], data_frame_b[add_features])
        self.assertIsInstance(join_results, result.ListResult)
        self.assertEqual(len(join_results.value), 1)
        component_results = join_results.value[0]
        self.assertIsInstance(component_results, result.ListResult)
        self.assertIsInstance(component_results.value[0], result.MappingResult)


if __name__ == '__main__':
    unittest.main()
