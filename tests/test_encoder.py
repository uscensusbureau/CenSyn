import numpy as np
import pandas as pd
import unittest

from censyn.encoder import Encoder, IdentityEncoder, EncodeProcessor, NumericalEncoder, OneHotEncoder

education = {'no HS': 1, 'HS': 2, 'some college': 3, 'Associate': 4, 'BS': 5, 'MS': 6, 'Phd': 7}
name = {'Bob': 1, 'Alice': 2, 'Joe': 3, 'Chris': 4}
year = {2000: 0, 2001: 1, 2002: 2, 2003: 3, 2004: 4, 2005: 5, 2006: 6, 2007: 7, 2008: 8, 2009: 9,
        2010: 10, 2011: 11, 2012: 12, 2013: 13, 2014: 14, 2015: 15, 2016: 16, 2017: 17, 2018: 18, 2019: 19}

test_df = pd.DataFrame({'name': ['Bob', 'Alice', 'Joe', 'Chris'],
                        'age': [28, 43, np.nan, 0],
                        'edu': ['BS', 'some college', 'HS', None],
                        'year': [2002, 2008, 2005, 2001],
                        'sex': ['m', 'f', 'm', None],
                        })
test_num_rows, test_num_columns = test_df.shape


class TestEncoder(unittest.TestCase):

    def test_no_indicator_with_none_nan_values(self) -> None:
        with self.assertRaises(ValueError):
            self.validate(NumericalEncoder(column='edu', mapping=education, indicator=False), 1)
            self.validate(IdentityEncoder(column='edu', indicator=False), 1)
        self.validate(IdentityEncoder(column='name', indicator=False), 1)
        self.validate(NumericalEncoder(column='name', mapping=name, alpha=0, indicator=False), 1)

    def test_encoder(self) -> None:
        self.validate(IdentityEncoder(column='name', inplace=True, indicator=True), 2)
        self.validate(IdentityEncoder(column='sex', indicator=True), 2)
        self.validate(IdentityEncoder(column='sex', indicator=True, inplace=True), 2)

    def test_numerical_encoder(self) -> None:
        with self.assertRaises(ValueError):
            NumericalEncoder(column='edu', mapping=education, alpha=-0.25)

        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.0, indicator=True), 2)
        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.25, indicator=True), 3)
        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.40, indicator=True), 4)
        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.0, inplace=True, indicator=True), 2)
        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.25, indicator=True, inplace=True), 3)
        self.validate(NumericalEncoder(column='edu', mapping=education, alpha=0.40, indicator=True, inplace=True), 4)
        self.validate(NumericalEncoder(column='edu', mapping=education, indicator=True, inplace=True), 4)
        self.validate(NumericalEncoder(column='year', mapping=year, alpha=0.0, indicator=False, inplace=True), 1)
        self.validate(NumericalEncoder(column='year', mapping=year, alpha=0.0, indicator=True, inplace=True), 2)
        self.validate(NumericalEncoder(column='year', mapping=year, alpha=0.25, indicator=True, inplace=True), 6)
        self.validate(NumericalEncoder(column='year', mapping=year, alpha=0.40, indicator=True, inplace=True), 9)
        self.validate(NumericalEncoder(column='year', mapping=year, indicator=True, inplace=True), 6)
        self.validate(NumericalEncoder(column='year', indicator=True, inplace=True), 3)
        self.validate(NumericalEncoder(column='edu', mapping={}, alpha=0.25, indicator=True, inplace=True), 2)
        self.validate(NumericalEncoder(column='edu', mapping={}, alpha=0.40, indicator=True, inplace=True), 2)
        self.validate(NumericalEncoder(column='name', mapping={}, alpha=0.40, indicator=True, inplace=True), 3)
        self.validate(NumericalEncoder(column='name', indicator=True, inplace=True), 3)

    def test_onehot_encoder(self) -> None:
        self.validate(OneHotEncoder(column='sex', mapping={'m': 'sex_male', 'f': 'sex_female', '': 'sex_other'},
                                    indicator=True), 4)
        self.validate(OneHotEncoder(column='sex', mapping={'m': 'sex_male', 'f': 'sex_female', '': 'sex_other'},
                                    indicator=True, inplace=True), 4)

    def validate(self, encoder: Encoder, num_cols: int):
        column = encoder.column
        test_index = pd.Index(test_df.columns).get_loc(column)
        t_df = test_df.copy()
        e_df = encoder.encode(t_df)
        if encoder.inplace:
            self.assertIsNone(e_df)
            e_df = t_df
            t_cols = num_cols + test_num_columns - 1
            if encoder.indicator:
                index = pd.Index(e_df.columns).get_loc(encoder.indicator_name)
                self.assertEqual(test_index, index, msg='indicator column index should be same as test index.')
        else:
            if encoder.indicator:
                index = pd.Index(e_df.columns).get_loc(encoder.indicator_name)
                self.assertEqual(0, index, msg='indicator column should have an index of 0.')
            t_cols = num_cols
        e_rows, e_columns = e_df.shape
        self.assertEqual(t_cols, e_columns)
        names = encoder.encode_names
        self.assertEqual(num_cols - encoder.indicator, len(names))
        self.assertEqual(num_cols, len(encoder.indicator_and_encode_names))
        if len(names) == 1:
            self.assertEqual([column], names)

        d_df = encoder.decode(e_df)
        if encoder.inplace:
            self.assertIsNone(d_df)
            d_df = e_df
            t_cols = test_num_columns
            index = pd.Index(d_df.columns).get_loc(column)
            self.assertEqual(test_index, index, msg='column index should be same as test index.')
        else:
            t_cols = 1
        d_rows, d_columns = d_df.shape
        self.assertEqual(t_cols, d_columns)
        self.assertTrue(t_df[column].equals(d_df[column]))

    def test_encode_processor(self) -> None:
        t_df = test_df.copy()

        processor = EncodeProcessor(report=True, inplace=False)
        processor.append_encoder(IdentityEncoder(column='age', indicator=True))
        processor.append_encoder(IdentityEncoder(column='name'))
        processor.append_encoder(NumericalEncoder(column='sex', mapping={'m': 1, 'f': 2}, indicator=True))
        out_df = processor.execute(t_df)
        out_rows, out_columns = out_df.shape
        config = processor.get_configuration()

        self.assertEqual(test_num_rows, out_rows)
        self.assertEqual(5, out_columns)

        report_df = processor.get_report_data()
        self.assertEqual(1, report_df['age_indicator'].loc['sum'])
        self.assertEqual(4, report_df['age'].loc['count'])
        self.assertEqual(4, report_df['name'].loc['unique'])

        p2 = EncodeProcessor(report=False, inplace=True)
        p2.set_configuration(config)
        out2_df = p2.execute(t_df)
        self.assertTrue(out_df.equals(out2_df))

    def test_encode_processor_inplace(self) -> None:
        t_df = test_df.copy()
        in_rows, in_columns = t_df.shape

        processor = EncodeProcessor(report=True)
        processor.inplace = True
        processor.append_encoder(IdentityEncoder(column='age', indicator=True, inplace=True))
        processor.append_encoder(IdentityEncoder(column='name'))
        processor.append_encoder(NumericalEncoder(column='sex', mapping={'m': 1, 'f': 2}, indicator=True,
                                                  inplace=False))
        out_df = processor.execute(t_df)
        self.assertIsNone(out_df)
        out_rows, out_columns = t_df.shape

        self.assertEqual(test_num_rows, out_rows)
        self.assertEqual(in_rows, out_rows)
        self.assertEqual(in_columns + 2, out_columns)

        report_df = processor.get_report_data()

        self.assertEqual(1, report_df['age_indicator'].loc['sum'])
        self.assertEqual(4, report_df['age'].loc['count'])
        self.assertEqual(4, report_df['name'].loc['unique'])
