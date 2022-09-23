import os
import unittest
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import censyn.datasources as ds
import tests
from censyn.encoder import Binner, NULL_INDEX, OTHER_INDEX
from censyn.encoder.binner import parse_tuple


class BinnerTest(unittest.TestCase):
    # Bunch of useful paths.
    _parent = Path(__file__).parent.parent
    _data_path = _parent / 'data'

    def test_parse_tuple_non_tuples(self) -> None:
        """Test for util.parse_tuple(), for non-tuple values."""
        test_values = [
            'string_value'
            '(with_beginning)parentheses',
            'with(middle)parentheses',
            'with(ending_parentheses)',
            1, 2, -100, np.inf,
            [], ["(in a list)"],
            {"random": "dictionary"},
            None
        ]
        for test_value in test_values:
            with self.subTest():
                self.assertEqual(test_value, parse_tuple(value=test_value, is_numeric=True),
                                 f'Value should not have been modified but was: {test_value}')

    def test_parse_tuple_valid_tuples(self) -> None:
        """Test for util.parse_tuple(), for valid tuple values."""
        test_tuples = [
            ['(-np.inf, np.inf)', (-np.inf, np.inf)],
            ['(1, 2)', (1, 2)],
            ['( 1, 2)', (1, 2)],
            ['(1, 2 )', (1, 2)],
            ['( 1, 2 )', (1, 2)],
            ['(1)', tuple([1])],
            ['(0, 12, 3)', (0, 12, 3)],
            ['( 0, 12, 3)', (0, 12, 3)],
            ['(0, 12, 3 )', (0, 12, 3)],
            ['( 0, 12, 3 )', (0, 12, 3)],
            ['( 0, 12, 3.0 )', (0, 12, 3)],
            ['( 0, 10000000, 10)', (0, 10000000, 10)],
            ['( -100000, -1)', (-100000, -1)],
            ['( -100000, 100000)', (-100000, 100000)],
            ['( -1234567890, 1234567890, 1234567890)', (-1234567890, 1234567890, 1234567890)]
        ]
        for test_tuple in test_tuples:
            with self.subTest():
                self.assertEqual(test_tuple[1], parse_tuple(value=test_tuple[0], is_numeric=True),
                                 f'Tuple mapping does not match expected expansion: {test_tuple}')

    def test_parse_tuple_invalid_tuples(self) -> None:
        """Test for Binner.clean_bin_mapping(), for valid tuple values."""
        test_tuples = [
            '(foo,bar)',
            '(1, foo)',
            '(foo, 1)',
            '(1.1., 0)',
            '()',
            '(0)',
            '(-1)',
            '(1.5)',
            '(2, 1)',
            '(2, 1, 1)',
            '(1, 2, 1.5)',
            '(1, 2, 0)',
            '(1, 2, -1)',
            '(1, 2, np.inf)',
            '(1,2,3,4)',
            '(1,2,3,4,5,6)',
            '(1, 2, 3, 4, 5, 6)',
            '(1,	2,	3,	4,	5,	6)',
            '(1,2,3)(1,2,3)',
            '(1,2,3),(1,2,3)'
        ]
        for test_tuple in test_tuples:
            with self.subTest():
                with self.assertRaises(ValueError):
                    parse_tuple(value=test_tuple, is_numeric=True)

    def get_data_file(self, file_name: str) -> Union[None, Path]:
        if tests.nas_connection_available():
            nas_file = tests.get_nas_file(file_name)
            if nas_file:
                if os.path.exists(str(nas_file)):
                    return nas_file
        local_file = self._data_path / file_name
        return local_file if os.path.exists(local_file) else None

    # Begin Bin tests
    def test_clean_bin_mapping_valid_non_tuple(self) -> None:
        """Test for Binner.clean_bin_mapping(), for non-tuple values."""
        test_mappings = [
            {},
            {'string_index1': 'string_value'},
            {1: 1},
            {2: np.inf},
            {3: 'string_value2'},
            {'string': []},
            {'string': ['one_value']},
            {
                'Western Europe': ['1', '3', '5', '8', '9', '11', '12', '20', '21', '22', '24', '26', '32',
                                   '40', '46', '49', '50', '51', '68', '77', '78', '82', '84', '87', '88',
                                   '89', '91', '94', '97', '98', '99'],
                'Eastern Europe': [100, 102, 103, 109, 111, 112, 114, 115, 122, 124, 125, 128, 129, 130, 131, 142, 144,
                                   146, 148, 152, 153, 154, 168, 169, 170, 171, 176, 177, 178, 179],
                'Test': ['1110XX', 111021]
            }
        ]
        for test_mapping in test_mappings:
            with self.subTest():
                self.assertEqual(Binner.clean_bin_mapping(is_numeric=True, mapping=test_mapping), test_mapping,
                                 f'Mapping should not have been modified but was: {test_mapping}')

    def test_clean_bin_mapping_valid_tuple(self) -> None:
        """Test for Binner.clean_bin_mapping(), for tuple values."""
        test_mappings = [
            [{'numpy': '(-np.inf, np.inf)'}, {'numpy': (-np.inf, np.inf)}],
            [{'range_of_values': '(0, 100)'}, {'range_of_values': (0, 100)}],
            [{'range_of_values': '(-100, 100)'}, {'range_of_values': (-100, 100)}],
            [{'range_of_values': '(0, np.inf)'}, {'range_of_values': (0, np.inf)}],
            [{'range_of_values': '(-np.inf, 0)'}, {'range_of_values': (-np.inf, 0)}],
            [{'pair': '(1, 2)'},   {'pair': (1, 2)}],
            [{'pair': '( 1, 2)'},  {'pair': (1, 2)}],
            [{'pair': '(1, 2 )'},  {'pair': (1, 2)}],
            [{'pair': '( 1, 2 )'}, {'pair': (1, 2)}],
            [
                {'triple': '(0, 12, 3)'},
                {'triple [0.0 4.0)': (0, 4), 'triple [4.0 8.0)': (4, 8), 'triple [8.0 12.0)': (8, 12)}
            ],
            [
                {'triple': '( 0, 12, 3)'},
                {'triple [0.0 4.0)': (0, 4), 'triple [4.0 8.0)': (4, 8), 'triple [8.0 12.0)': (8, 12)}
            ],
            [
                {'triple': '(0, 12, 3 )'},
                {'triple [0.0 4.0)': (0, 4), 'triple [4.0 8.0)': (4, 8), 'triple [8.0 12.0)': (8, 12)}
            ],
            [
                {'triple': '( 0, 12, 3 )'},
                {'triple [0.0 4.0)': (0, 4), 'triple [4.0 8.0)': (4, 8), 'triple [8.0 12.0)': (8, 12)}
            ],
            [
                {'logs': 'log(1, 10, 3)'},
                {'logs [1.0 2.2)': (1.0, 2.2), 'logs [2.2 4.6)': (2.2, 4.6), 'logs [4.6 10.0)': (4.6, 10.0)}
            ],
            [
                {'logs': 'log(-10, -1, 3)'},
                {'logs [-10.0 -4.6)': (-10.0, -4.6), 'logs [-4.6 -2.2)': (-4.6, -2.2), 'logs [-2.2 -1.0)': (-2.2, -1.0)}
            ],
            [
                {'floats': '( 0, 10, 3 )'},
                {'floats [0.0 3.3)': (0, round(3+1/3, 1)),
                 'floats [3.3 6.7)': (round(3+1/3, 1), round(6+2/3, 1)),
                 'floats [6.7 10.0)': (round(6+2/3, 1), 10)}
            ],
            [
                {'a bit': '( 0, 10000, 10 )'},
                {
                    'a bit [0.0 1000.0)':     (0.0, 1000.0),
                    'a bit [1000.0 2000.0)':  (1000.0, 2000.0),
                    'a bit [2000.0 3000.0)':  (2000.0, 3000.0),
                    'a bit [3000.0 4000.0)':  (3000.0, 4000.0),
                    'a bit [4000.0 5000.0)':  (4000.0, 5000.0),
                    'a bit [5000.0 6000.0)':  (5000.0, 6000.0),
                    'a bit [6000.0 7000.0)':  (6000.0, 7000.0),
                    'a bit [7000.0 8000.0)':  (7000.0, 8000.0),
                    'a bit [8000.0 9000.0)':  (8000.0, 9000.0),
                    'a bit [9000.0 10000.0)': (9000.0, 10000.0)
                }
            ]
        ]
        for test_mapping in test_mappings:
            with self.subTest():
                self.assertEqual(test_mapping[1], Binner.clean_bin_mapping(is_numeric=True, mapping=test_mapping[0]),
                                 f'Tuple mapping does not match expected expansion: {test_mapping}')

    def test_clean_bin_mapping_invalid(self) -> None:
        """Test for Binner.clean_bin_mapping(), for invalid input."""
        mappings = [
            {'range_of_values': '(foo,bar)'},
            {'range_of_values': '(1, foo)'},
            {'range_of_values': '(foo, 1)'},
            {'range_of_values': '(1.1., 0)'},
            {'range_of_values': '()'},
            {'range_of_values': '(1)'},
            {'range_of_values': '(1,2,0)'},
            {'range_of_values': '(1,2,-1)'},
            {'range_of_values': '(1,2,np.inf)'},
            {'range_of_values': '(1,2,3,4)'},
            {'range_of_values': '(1,2,3,4,5,6)'},
            {'range_of_values': '(1, 2, 3, 4, 5, 6)'},
            {'range_of_values': '(1,	2,	3,	4,	5,	6)'},
            {'range_of_values': '(1,2,3)(1,2,3)'},
            {'range_of_values': '(1,2,3),(1,2,3)'},
            {'range_of_values': '(1,)'},
            {'range_of_values': '(,1)'},
            {'range_of_values': '(1..0,2)'},
            {'range_of_values': '(1.0,2..0)'},
            {'range_of_values': '(1,(2))'},
            {'range_of_values': '((1),2)'},
            {'range_of_values': '((1,2))'},
        ]
        for mapping in mappings:
            with self.subTest():
                with self.assertRaises(ValueError):
                    Binner.clean_bin_mapping(is_numeric=True, mapping=mapping)

    def test_bin(self) -> None:
        """Test for Binner.bin()"""
        # Technically, the string labels don't matter, they could all be anything, including duplicates.
        # They're only included here to make it easy to test they are binned with the correct indices below.
        bin_list = {
            "0": -np.inf,
            "1": (-20, -10),
            "2": (-10, 0),
            "3": 0,
            "4": (0.000001, 10),
            "5": (10, 20),
            "6": (100, 200),
            "7": 300,
            "8": 300.1,
            "9": (1000.1, 999999999.9),
            "10": '500',
            "11": '500',
            "12": [500, 501, '502'],
            "13": '500'
        }

        # Tuples here are (value, index), with value being the value to be binned using apply_bin, and
        # index is the expected index, based on the bin_list above
        values_to_bins = [
            (-np.inf, 2),
            (np.nan, NULL_INDEX),
            (None, NULL_INDEX),
            (-100, OTHER_INDEX),
            (-20, 3),
            (-15, 3),
            (-10, 4),
            (0, 5),
            (0.1, 6),
            (19.99999999, 7),
            (20, OTHER_INDEX),
            (99.999999999, OTHER_INDEX),
            (100, 8),
            (100.0, 8),
            (100.000, 8),
            (300, 9),
            (300.1, 10),
            (300.00001, OTHER_INDEX),
            (300.09999, OTHER_INDEX),
            (500, 15),
            ('500', 15),
            ('501', 14),
            ('502', 1)
        ]
        # As data type float
        binner = Binner(is_numeric=True, mapping=bin_list)
        for value, res in values_to_bins:
            with self.subTest():
                test_s = pd.Series(data=[value], dtype=float)
                out_s = binner.bin(in_s=test_s)
                expected_s = pd.Series(data=res, dtype=int)
                pd.testing.assert_series_equal(out_s, expected_s)

    def test_percentile_bin(self) -> None:
        params = [
            ({'percentile': 'percentile(4)'}, [2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
            ({'percentile': 'percentile(10)'}, [2, 2, 5, 5, 5, 6, 7, 8, 9, 10, 11, 11]),
            ({'percentile': 'percentile(0, 10, 5)'}, [2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6]),
            ({'percentile': 'percentile(4, 10, 2)'}, [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        ]
        test_s = pd.Series(data=[0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9])
        for mapping, res in params:
            with self.subTest():
                binner = Binner(is_numeric=True, mapping=mapping)
                out_s = binner.bin(in_s=test_s)
                expected_s = pd.Series(data=res, dtype=int)
                pd.testing.assert_series_equal(out_s, expected_s)

    def test_bin_full_dataset(self) -> None:
        """
        The idea behind this test is that our feature.json should be able to appropriately bin
        all the values in a real dataset - in this case, the VT persons file from 2016.
        Since the dataset is cleaned, all of our bins should be able to cover all the values in the dataset.
        Therefore, if any binned values come back with the OTHER_INDEX
        (indicating that the value was not null or NaN and yet was still unable to be binned)
        then it means there is something wrong.

        Failing of this test indicates that there is either something wrong with the features.json or
        with the binning implementation and requires further investigation.
        """
        data_file = self.get_data_file('personsIL2016.parquet')
        if not data_file:
            raise unittest.SkipTest(tests.NAS_UNAVAILABLE_MESSAGE)
        df = ds.ParquetDataSource(path_to_file=str(data_file)).to_dataframe()

        conf_path = Path(__file__).parent.parent / 'conf'
        self._json_definition_path = conf_path / 'features_PUMS-P.json'

        other_values = {
            'PWGTP': 3
        }
        json_source = ds.FeatureJsonFileSource(path_to_file=self._json_definition_path)
        features = json_source.feature_definitions
        for feature in features:
            if feature.binner is not None and feature.feature_name in df.columns:
                transformed_data_frame: pd.DataFrame = feature.transform(df)
                other_cnt = len(transformed_data_frame.query(f'{feature.feature_name} == {OTHER_INDEX}'))
                if feature.feature_name in other_values.keys():
                    self.assertEqual(other_values[feature.feature_name], other_cnt,
                                     f'For feature {feature.feature_name}, '
                                     f'there are some values that could not be binned.')
                else:
                    self.assertEqual(0, other_cnt, f'For feature {feature.feature_name}, '
                                                   f'there are some values that could not be binned.')


if __name__ == '__main__':
    unittest.main()
