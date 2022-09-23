import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from censyn.datasources import FeatureJsonFileSource
from censyn.filters import *
from censyn.filters.feature_filter import FilterFirstN, FilterLastN, FilterFeaturesBefore, FilterRandomFeatures, \
    FilterByFeatureType, FeatureType, FilterFeatureByRegex
from censyn.filters.header_filter import HeaderFilter


test_df = pd.DataFrame({'name': ['Bob', 'Alice', None, 'Chris', 'Henry', 'Sue'],
                        'age': [28, 55, 16, 57, np.nan, 69],
                        'sex': ['m', 'f', 'm', 'm', 'm', 'f']
                        })
test_number_rows, test_columns = test_df.shape
# Using variables in asserts to test expected values more clearly
num_55 = 1
num_less_than_55 = 2
num_greater_than_55 = 2
num_nan = 1

random_count = 4


class TestFilters(unittest.TestCase):
    def test_filter(self) -> None:
        identity_filter = Filter()
        modified_df = identity_filter.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, test_number_rows, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        identity_filter_neg = Filter(negate=True)
        modified_neg_df = identity_filter_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, 0,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_random(self) -> None:
        filter_proportion = RandomFilter(proportion=0.25, seed=1)
        modified_df = filter_proportion.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, round(test_number_rows * 0.25),
                         'Modified number of rows is equal to number of rows')
        self.assertEqual(modified_df.index[0], 2, 'Index is not correct')
        self.assertEqual(modified_df.index[1], 1, 'Index is not correct')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_proportion_neg = RandomFilter(proportion=0.25, negate=True)
        modified_neg_df = filter_proportion_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, round(test_number_rows * (1 - 0.25)),
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_number = RandomFilter(count=random_count)
        modified_df = filter_number.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, random_count, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_number_neg = RandomFilter(count=random_count, negate=True)
        modified_neg_df = filter_number_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - random_count,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_index(self) -> None:
        test_indexes = [1, 2, 4, 5]
        filter_index = IndexFilter(indexes=test_indexes)
        modified_df = filter_index.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, len(test_indexes), 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_index_neg = IndexFilter(indexes=test_indexes, negate=True)
        modified_neg_df = filter_index_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - len(test_indexes),
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_index_range(self) -> None:
        filter_range = IndexRangeFilter(start=2, end=4)
        modified_df = filter_range.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 4 - 2, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_range_neg = IndexRangeFilter(start=2, end=4, negate=True)
        modified_neg_df = filter_range_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - (4 - 2),
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_equals(self) -> None:
        self.assertEqual(num_55 + num_less_than_55 + num_greater_than_55 + num_nan, test_number_rows,
                         'Testing values do not add up to total number of rows. Check your testing values.')

        filter_col = ColumnEqualsFilter(header='age', value=55)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_55, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsFilter(header='age', value=55, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_less_than_55 + num_greater_than_55 + num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnEqualsFilter(header='age', value=np.nan)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_nan, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsFilter(header='age', value=np.nan, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_less_than_55 + num_55 + num_greater_than_55,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnEqualsFilter(header='name', value='Chris')
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 1, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsFilter(header='name', value='Chris', negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - 1,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnEqualsFilter(header='name', value=None)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 1, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsFilter(header='name', value=None, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - 1,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_equals_list(self) -> None:
        filter_col = ColumnEqualsListFilter(header='name', value=['Alice', 'Chris', 'Henry'])
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 3, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsListFilter(header='name', value=['Alice', 'Chris', 'Henry'], negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, 3,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')
        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnEqualsListFilter(header='name', value=['Alice', 'Henry', None])
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 3, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsListFilter(header='name', value=['Alice', 'Henry', None], negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, 3,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')
        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnEqualsListFilter(header='age', value=[16, np.nan, 69])
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 3, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnEqualsListFilter(header='age', value=[16, np.nan, 69], negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, 3,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')
        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_less_than(self) -> None:
        filter_col = ColumnLessThanFilter(header='age', value=55)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_less_than_55, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnLessThanFilter(header='age', value=55, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_55 + num_greater_than_55 + num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnLessThanFilter(header='age', value=np.nan)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 0, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnLessThanFilter(header='age', value=np.nan, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_less_than_equal(self) -> None:
        filter_col = ColumnLessThanEqualFilter(header='age', value=55)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_less_than_55 + num_55,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnLessThanEqualFilter(header='age', value=55, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_greater_than_55 + num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_greater_than(self) -> None:
        filter_col = ColumnGreaterThanFilter(header='age', value=55)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_greater_than_55,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnGreaterThanFilter(header='age', value=55, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_less_than_55 + num_55 + num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_greater_than_equal(self) -> None:
        filter_col = ColumnGreaterThanEqualFilter(header='age', value=55)
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_greater_than_55 + num_55,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnGreaterThanEqualFilter(header='age', value=55, negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, num_less_than_55 + num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

    def test_column_is_null(self) -> None:
        filter_col = ColumnIsnullFilter(header='age')
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, num_nan, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnIsnullFilter(header='age', negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - num_nan,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')

        filter_col = ColumnIsnullFilter(header='name')
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 1, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_col_neg = ColumnIsnullFilter(header='name', negate=True)
        modified_neg_df = filter_col_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows - 1,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, test_columns, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_rows + modified_neg_rows, test_number_rows,
                         'The modified rows should sum to the number of rows')
    def test_expression(self) -> None:
        filter_col = ExpressionFilter(expr="sex == 'm' and age < 50")
        modified_df = filter_col.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, 2, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

    def test_sequential(self) -> None:
        filters = [ColumnLessThanEqualFilter(header='age', value=55),
                   IndexRangeFilter(start=1, end=4),
                   RandomFilter(proportion=0.33)]
        cur_df = test_df
        for f in filters:
            prev_rows, prev_columns = cur_df.shape
            cur_df = f.execute(cur_df)
            modified_rows, modified_columns = cur_df.shape
            self.assertLess(modified_rows, prev_rows, 'Modified number of rows is not less than number of rows')
            self.assertEqual(modified_columns, test_columns, 'Modified number of columns is not equal to test columns')

    def test_header(self) -> None:
        filter_header = HeaderFilter(headers=['age'])
        modified_df = filter_header.execute(test_df)
        modified_rows, modified_columns = modified_df.shape
        self.assertEqual(modified_rows, test_number_rows, 'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_columns, 1, 'Modified number of columns is not equal to test columns')

        # Negate
        filter_header_neg = HeaderFilter(headers=['age'], negate=True)
        modified_neg_df = filter_header_neg.execute(test_df)
        modified_neg_rows, modified_neg_columns = modified_neg_df.shape
        self.assertEqual(modified_neg_rows, test_number_rows,
                         'Modified number of rows is not equal to number of rows')
        self.assertEqual(modified_neg_columns, 2, 'Modified number of columns is not equal to test columns')

        self.assertEqual(modified_columns + modified_neg_columns, test_columns,
                         'The modified columns should sum to the number of test columns')


class TestFilterFeature(unittest.TestCase):
    __test__ = True

    def setUp(self) -> None:
        conf_path = Path(__file__).resolve().parent.parent / 'conf'
        assets_path = Path(__file__).resolve().parent.parent / 'tests' / 'assets'
        self._json_definition_path = conf_path / 'features_PUMS-P.json'
        self._definition_path_copy = assets_path / 'features_copy.json'
        self._json_schema_path = conf_path / 'features_PUMS-P_schema.json'
        self._malformed_json = assets_path / 'ACS_15_5YR_DP02_socCh.csv'  # Borrowed

        with open(self._json_definition_path) as w:
            self._decoded_json = json.loads(w.read())

        json_source = FeatureJsonFileSource(path_to_file=self._json_definition_path)

        self._features = json_source.feature_definitions

    def tearDown(self) -> None:
        pass

    def test_firstN(self) -> None:
        feature_count = 5

        header_list = [x.feature_name for x in self._features]

        # First N Features all you should have to do is provide a count.
        first_n = FilterFirstN(feature_count=feature_count)
        returned_features_first = first_n.execute(self._features)

        self.assertEqual(len(returned_features_first), feature_count)

        first_n_expected = header_list[:feature_count]
        for idx, val in enumerate(returned_features_first):
            self.assertEqual(first_n_expected[idx], val.feature_name)

        first_n_negate = FilterFirstN(feature_count=feature_count, negate=True)
        returned_features_first_negate = first_n_negate.execute(self._features)

        self.assertEqual(len(returned_features_first_negate), len(self._features) - feature_count)

        first_n_expected_negate = header_list[feature_count:]
        for idx, val in enumerate(returned_features_first_negate):
            self.assertEqual(first_n_expected_negate[idx], val.feature_name)

    def test_lastN(self) -> None:
        # Last N Features
        feature_count = 5
        header_list = [x.feature_name for x in self._features]

        last_n = FilterLastN(feature_count=feature_count)
        returned_features_last = last_n.execute(self._features)

        self.assertEqual(len(returned_features_last), feature_count)

        last_n_expected = ['SFN', 'SFR', 'SOCP', 'VPS', 'WAOB']
        for idx, val in enumerate(returned_features_last):
            self.assertEqual(last_n_expected[idx], val.feature_name)

        last_n_negate = FilterLastN(feature_count=feature_count, negate=True)
        returned_features_last_negate = last_n_negate.execute(self._features)

        self.assertEqual(len(returned_features_last_negate), len(self._features) - feature_count)

        last_n_expected_negate = header_list[:len(self._features) - feature_count]
        for idx, val in enumerate(returned_features_last_negate):
            self.assertEqual(last_n_expected_negate[idx], val.feature_name)

    def test_feature_before_and_after(self) -> None:
        feature_name_before = 'PUMA'
        feature_name_after = 'SFR'
        expected_returned_features_before = 3
        expected_returned_features_after = 3

        # This will provide you with the features before or after a given feature.
        filter_before = FilterFeaturesBefore(feature_name=feature_name_before)
        returned_features_before = filter_before.execute(self._features)

        self.assertEqual(len(returned_features_before), expected_returned_features_before)

        # Example of after you just provide negate=True
        filter_after = FilterFeaturesBefore(feature_name=feature_name_after, negate=True)
        returned_features_after = filter_after.execute(self._features)

        self.assertEqual(len(returned_features_after), expected_returned_features_after)

    def test_random_feature_filter(self) -> None:
        percentage = 0.5
        number_of_features = 100
        expected_returned_features_percent = 64
        expected_returned_features_count = 100

        # Random filter of features both percentage and number
        filter_random_percentage = FilterRandomFeatures(percent=percentage)
        returned_features_percent = filter_random_percentage.execute(self._features)

        self.assertEqual(len(returned_features_percent), expected_returned_features_percent)

        filter_random_count = FilterRandomFeatures(count=number_of_features)
        returned_features_count = filter_random_count.execute(self._features)

        self.assertEqual(len(returned_features_count), expected_returned_features_count)

    def test_filter_by_feature_type(self) -> None:
        expected_object = 24
        expected_integer = 102
        expected_floating_point = 3
        expected_negate_floating_point = len(self._features) - expected_floating_point

        # Filter by Type example.
        filter_by_type_obj = FilterByFeatureType(feature_type=FeatureType.obj)
        returned_object = filter_by_type_obj.execute(self._features)
        self.assertEqual(len(returned_object), expected_object)

        # Filter by Type example.
        filter_by_type = FilterByFeatureType(feature_type=FeatureType.floating_point)
        returned_floating_point = filter_by_type.execute(self._features)

        self.assertEqual(len(returned_floating_point), expected_floating_point)

        # Filter by Type example.
        filter_by_type_integer = FilterByFeatureType(feature_type=FeatureType.integer)
        returned_floating_point_integer = filter_by_type_integer.execute(self._features)

        self.assertEqual(len(returned_floating_point_integer), expected_integer)

        # Negated Filter by Type
        filter_by_type_negate = FilterByFeatureType(feature_type=FeatureType.floating_point, negate=True)
        returned_floating_point_negate = filter_by_type_negate.execute(self._features)

        self.assertEqual(len(returned_floating_point_negate), expected_negate_floating_point)

    def test_filter_regex(self) -> None:
        expected_filter_by_regex = 13
        expected_filter_by_regex_negate = len(self._features) - expected_filter_by_regex

        # Regex Example
        filter_by_regex = FilterFeatureByRegex(regex=r'RAC*')
        returned_filter_by_regex = filter_by_regex.execute(self._features)

        self.assertEqual(len(returned_filter_by_regex), expected_filter_by_regex)

        # Negated regex example
        filter_by_regex_negate = FilterFeatureByRegex(regex=r'RAC*', negate=True)
        returned_filter_by_regex_negate = filter_by_regex_negate.execute(self._features)

        self.assertEqual(len(returned_filter_by_regex_negate), expected_filter_by_regex_negate)

    def test_filter_dataframe(self) -> None:
        expected_dataframe_shape = (100, 0)
        expected_dataframe_shape_negate = (100, len(self._features))

        header_list = [x.feature_name for x in self._features]

        filter_feature = FilterFeatureByRegex(regex=r'wgt*')
        mod_features = filter_feature.execute(self._features)
        mod_headers = [f.feature_name for f in mod_features]
        filter_dataframe = HeaderFilter(headers=mod_headers)
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(header_list))), columns=header_list)
        returned_filtered_dataframe = filter_dataframe.execute(df=df)

        x, y = returned_filtered_dataframe.shape

        self.assertEqual(expected_dataframe_shape, (x, y))

        filter_feature = FilterFeatureByRegex(regex=r'wgt*')
        mod_features = filter_feature.execute(self._features)
        mod_headers = [f.feature_name for f in mod_features]
        filter_dataframe_negate = HeaderFilter(headers=mod_headers, negate=True)
        df_negate = pd.DataFrame(np.random.randint(0, 100, size=(100, len(header_list))), columns=header_list)
        returned_filtered_dataframe_negate = filter_dataframe_negate.execute(df=df_negate)

        x, y = returned_filtered_dataframe_negate.shape

        self.assertEqual(expected_dataframe_shape_negate, (x, y))

        # Double negate Test. We negate both the filter that is provided and the FilterDataFrame Class
        filter_feature = FilterFeatureByRegex(regex=r'wgt*', negate=True)
        mod_features = filter_feature.execute(self._features)
        mod_headers = [f.feature_name for f in mod_features]
        filter_dataframe_double_negate = HeaderFilter(headers=mod_headers, negate=True)
        df_negate = pd.DataFrame(np.random.randint(0, 100, size=(100, len(header_list))), columns=header_list)
        returned_filtered_dataframe_double_negate = filter_dataframe_double_negate.execute(df=df_negate)

        x, y = returned_filtered_dataframe_double_negate.shape

        self.assertEqual(expected_dataframe_shape, (x, y))
