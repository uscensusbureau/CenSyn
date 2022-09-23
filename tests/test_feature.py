import unittest

import numpy as np
import pandas as pd

from censyn.checks import ParseError
from censyn.encoder import Binner, OTHER_INDEX, NULL_INDEX
from censyn.features import Feature, FeatureType, ModelChoice, ModelSpec


class FeatureTest(unittest.TestCase):
    def test_feature_format(self) -> None:
        """Test of the feature format is valid"""
        with self.assertRaises(ValueError):
            Feature(feature_name='test1', feature_type=FeatureType.obj, feature_format="padleft(PUMA, 5, '0')")

        with self.assertRaises(ParseError):
            Feature(feature_name='test2', feature_type=FeatureType.obj, feature_format="padleft(test2, 5)")

    def test_feature_calculated_expression(self) -> None:
        with self.assertRaises(ValueError):
            Feature(feature_name='test1', feature_type=FeatureType.integer,
                    model_type=ModelSpec(model=ModelChoice.CalculateModel, model_params={'expr': 'test2'}),
                    dependencies=['test3'])

        with self.assertRaises(ParseError):
            Feature(feature_name='test2', feature_type=FeatureType.integer,
                    model_type=ModelSpec(model=ModelChoice.CalculateModel, model_params={'expr': 'test2 +'}),
                    dependencies=['test2'])

    @staticmethod
    def test_Feature_transform() -> None:
        # define list of data frames to transform
        col_names = ['categorical', 'scalar', 'all_nan', 'zero_row', 'one_row']

        data_frame_a = pd.DataFrame(columns=col_names)
        data_frame_a['scalar'] = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 10.0, 20.0, 30.0, 40.0, 50.0])

        data_frame_b = pd.DataFrame(columns=col_names)
        data_frame_b['scalar'] = pd.Series([1.0, 2.0, 3.0, 6.0, np.nan, 10.0, 20.0, 30.0, 40.0, 50.0])

        scalar_binner = Binner(is_numeric=True,
                               mapping={"1": 1.0, "2": (2.0, 3.5), "3": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scalar_feature = Feature(feature_name='scalar', feature_type=FeatureType.floating_point, binner=scalar_binner)

        # specify expected binned values for data frame a
        data_frame_a_binned_expected = pd.DataFrame(columns=col_names)
        data_frame_a_binned_expected['scalar'] = pd.Series([2, 3, 3, OTHER_INDEX, NULL_INDEX, 4, 4, 4, 4, 4],
                                                           name='scalar', dtype=int)
        data_frame_a_binned_expected['labels'] = pd.Series(["1", "2", "2", "Other", "Null", "3", "3", "3", "3", "3"])

        # specify expected binned values for data frame b
        data_frame_b_binned_expected = pd.DataFrame(columns=col_names)
        data_frame_b_binned_expected['scalar'] = pd.Series([2, 3, 3, OTHER_INDEX, NULL_INDEX, 4, 4, 4, 4, 4],
                                                           name='scalar', dtype=int)
        data_frame_b_binned_expected['labels'] = pd.Series(["1", "2", "2", "Other", "Null", "3", "3", "3", "3", "3"])

        # obtain actual binned values for data frames a and b
        data_frame_a_binned_actual = scalar_feature.transform(data_frame_a)
        data_frame_b_binned_actual = scalar_feature.transform(data_frame_b)

        data_frame_a_binned_actual['labels'] = scalar_binner.bins_to_labels(data_frame_a_binned_actual['scalar'])
        data_frame_b_binned_actual['labels'] = scalar_binner.bins_to_labels(data_frame_b_binned_actual['scalar'])

        # assert that both data frames are binned correctly.
        pd.testing.assert_series_equal(data_frame_a_binned_actual['scalar'], data_frame_a_binned_expected['scalar'])
        pd.testing.assert_series_equal(data_frame_a_binned_actual['labels'], data_frame_a_binned_expected['labels'])
        pd.testing.assert_series_equal(data_frame_b_binned_actual['scalar'], data_frame_b_binned_expected['scalar'])
        pd.testing.assert_series_equal(data_frame_b_binned_actual['labels'], data_frame_b_binned_expected['labels'])
