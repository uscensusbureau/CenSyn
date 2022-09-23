import unittest
import pandas as pd
import numpy as np

from censyn.models import models, CalculateModel, DecisionTreeModel, DecisionTreeRegressorModel, HierarchicalModel
from censyn.features import feature
from censyn.encoder import IdentityEncoder, NumericalEncoder


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_NoopModel_synthesize(self) -> None:
        # Set up the Feature and Model
        encoder = NumericalEncoder(column='target', mapping={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
                                   indicator=True, inplace=True)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                         encoder=encoder)
        noop_model = models.NoopModel(target_feature=target_feature)

        # Set up the data
        test_df = pd.DataFrame(columns=['categorical', 'target', 'all_nan', 'zero_row', 'one_row'])
        test_df['target'] = pd.Series(['a', 'b', 'a', 'c', 'e', 'd', 'e', 'b'])
        data_frame = test_df.copy()
        encoder.encode(in_df=data_frame)

        # train and synthesize
        noop_model.train(None, test_df['target'], None)
        synthesize_result = noop_model.synthesize(None)
        pd.testing.assert_series_equal(synthesize_result, test_df['target'])

    def test_RandomModel_synthesize(self) -> None:
        # Set up the Feature and Model
        encoder = IdentityEncoder(column='target', indicator=True, inplace=True)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                         encoder=encoder)
        rand_model = models.RandomModel(target_feature)

        # Set up the data
        test_df = pd.DataFrame(columns=['categorical', 'target', 'all_nan', 'zero_row', 'one_row'])
        test_df['target'] = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 10.0, 20.0, 30.0, 40.0, 50.0])
        weights = pd.Series(data=[1, 1, 1, 20, 1, 20, 1, 1, 20, 1], index=test_df.index)
        data_frame = test_df.copy()
        encoder.encode(in_df=data_frame)

        # train and synthesize
        rand_model.train(None, test_df['target'], weights)
        synthesize_result = rand_model.synthesize(None)
        self.assertEqual(synthesize_result.size, 10)

        # check each item in rand_result is one of the valid possible values
        t1 = test_df.dropna(axis=0, subset=['target'])
        for val in synthesize_result:
            if not np.isnan(val):  # can't compare a nan against a nan and get expected result
                params_val = t1.iloc[(t1['target'] - val).abs().argsort()[:]]['target'].values
                self.assertAlmostEqual(val, params_val[0])

    def test_DecisionTreeModel_synthesize(self) -> None:
        # Set up the Features and Model
        encoder = NumericalEncoder(column='target', mapping={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
                                   indicator=True, inplace=True)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.obj,
                                         encoder=encoder, dependencies=['predictor_a', 'predictor_b'])
        model = DecisionTreeModel(endpoints=['a'],
                                  sklearn_model_params={'max_depth': 10, 'criterion': 'entropy',
                                                        'min_impurity_decrease': 1e-5},
                                  target_feature=target_feature)

        # Set up the data
        col_names = ['predictor_a', 'predictor_b']
        test_df = pd.DataFrame(columns=col_names)
        test_df['predictor_a'] = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        test_df['predictor_b'] = pd.Series([10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
        data_frame = test_df.copy()
        target_series = pd.Series(['a', 'b', 'a', 'c', 'e', 'd', 'e', 'b', 'a', 'b'], name='target')
        weights = pd.Series(data=[1, 1, 1, 20, 1, 20, 1, 1, 20, 1], index=test_df.index)

        # train and synthesize
        model.train(data_frame, target_series, weights)
        synthesize_result = model.synthesize(test_df)
        self.assertEqual(synthesize_result.size, 10)

        target_series = pd.Series([2.0, 3.4, 2.0, 2.8, 2.9, 3.9, 3.5, 3.4, 2.0, 1.9], name='target')
        weights = pd.Series(data=[1, 1, 1, 20, 1, 20, 1, 1, 20, 1], index=test_df.index)

        # train and synthesize
        model.train(data_frame, target_series, weights)
        synthesize_result = model.synthesize(test_df)
        self.assertEqual(synthesize_result.size, 10)

    def test_DecisionTreeRegressorModel_synthesize(self) -> None:
        # Set up the Features and Model
        encoder = IdentityEncoder(column='target', indicator=False)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                         encoder=encoder,
                                         dependencies=['predictor_a', 'predictor_b'])

        model = DecisionTreeRegressorModel(endpoints=[0, 10],
                                           sklearn_model_params={'max_depth': 10, 'criterion': 'squared_error',
                                                                 'min_impurity_decrease': 1e-5},
                                           target_feature=target_feature)
        col_names = ['predictor_a', 'predictor_b']
        test_df = pd.DataFrame(columns=col_names)
        test_df['predictor_a'] = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        test_df['predictor_b'] = pd.Series([10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
        data_frame = test_df.copy()
        target_series = pd.Series([0, 0, 1, 1, 2, 2, 2, 2, 3, 4], name='target')

        # train and synthesize
        model.train(data_frame, target_series, None)
        synthesize_result = model.synthesize(test_df)
        self.assertEqual(synthesize_result.size, 10)

    def test_HierarchicalModel_train(self) -> None:
        # set up
        encoder = NumericalEncoder(column='target',
                                          mapping={'a1': 1, 'a2': 2, 'a3': 3, 'b1': 4, 'b2': 5, 'b3': 6,
                                                   'c1': 7, 'c2': 8},
                                          indicator=False)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                         encoder=encoder,
                                         dependencies=['predictor_a', 'predictor_b'])
        hierarchy_map = {'a1': ('a', '1'), 'a2': ('a', '2'), 'a3': ('a', '3'), 'b1': ('b', '1'), 'b2': ('b', '2'),
                         'b3': ('b', '2'), 'c1': ('c', '1'), 'c2': ('c', '2')}
        model = HierarchicalModel(hierarchy_map=hierarchy_map, target_feature=target_feature)

        # Set up the data
        col_names = ['predictor_a', 'predictor_b']
        test_df = pd.DataFrame(columns=col_names)
        test_df['predictor_a'] = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        test_df['predictor_b'] = pd.Series([10, 20, 10, 20, 10, 20, 20, 30, 10, 20])
        data_frame = test_df.copy()
        target_series = pd.Series(['a1', 'b2', 'a2', 'c2', 'a3', 'b1', 'b3', 'c1', 'a1', 'b1'], name='target')

        # train and synthesize
        model.train(data_frame, target_series, None)
        synthesize_result = model.synthesize(test_df)
        self.assertEqual(synthesize_result.size, 10)

    def test_HierarchicalModel_mapping(self) -> None:
        with self.assertRaises(ValueError):
            encoder = NumericalEncoder(column='target',
                                       mapping={'a1': 1, 'a2': 2, 'a3': 3, 'b1': 4, 'b2': 5, 'b3': 6,
                                                'c1': 7, 'c2': 8},
                                       indicator=False)
            target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                             encoder=encoder,
                                             dependencies=['predictor_a', 'predictor_b'])
            hierarchy_map = {'a1': ('a', '1'), 'a2': ('a', '2'), 'b1': ('b', '1'), 'b2': ('b', '2'),
                             'b3': ('b', '2'), 'c1': ('c', '1'), 'c2': ('c', '2')}
            model = HierarchicalModel(hierarchy_map=hierarchy_map, target_feature=target_feature)

            # Set up the data
            col_names = ['predictor_a', 'predictor_b']
            test_df = pd.DataFrame(columns=col_names)
            test_df['predictor_a'] = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
            test_df['predictor_b'] = pd.Series([10, 20, 10, 20, 10, 20, 20, 30, 10, 20])
            data_frame = test_df.copy()
            target_series = pd.Series(['a1', 'b2', 'a2', 'c2', 'a3', 'b1', 'b3', 'c1', 'a1', 'b1'], name='target')

            # train and synthesize
            model.train(data_frame, target_series, None)

    def test_CalculateModel_synthesize(self) -> None:
        # Set up the Features and Model
        encoder = NumericalEncoder(column='target', mapping={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
                                   indicator=True, inplace=True)
        target_feature = feature.Feature(feature_name='target', feature_type=feature.FeatureType.floating_point,
                                         encoder=encoder, dependencies=['predictor_a', 'predictor_b'])
        calc_model = CalculateModel(target_feature=target_feature,
                                    expr="if(predictor_b < 12 then 1 else predictor_a + 2)")

        # Set up the data
        col_names = ['predictor_a', 'predictor_b']
        test_df = pd.DataFrame(columns=col_names)
        test_df['predictor_a'] = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        test_df['predictor_b'] = pd.Series([10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
        data_frame = test_df.copy()
        target_series = pd.Series(['a', 'b', 'a', 'c', 'e', 'd', 'e', 'b', 'a', 'b'], name='target')

        # train and synthesize
        calc_model.train(data_frame, target_series, None)
        synthesize_result = calc_model.synthesize(data_frame)

        test_series = pd.Series([1, 1, 1, 1, 1, 5, 6, 6, 7, 7], name='target')
        pd.testing.assert_series_equal(synthesize_result, test_series)
