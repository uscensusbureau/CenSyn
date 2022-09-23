import unittest

import tests

from censyn.analytics import Analytic, Assertion
from censyn.datasources import ParquetDataSource
from censyn.results import BooleanResult


class AnalyticTest(unittest.TestCase):

    def setUp(self):
        if not tests.nas_connection_available():
            raise unittest.SkipTest(tests.NAS_UNAVAILABLE_MESSAGE)
        self.df_a = ParquetDataSource(path_to_file=str(tests.get_nas_file('original-persons-IL-2016.parquet'))).to_dataframe()
        self.df_a.name = 'df_a'
        self.df_b = ParquetDataSource(path_to_file=str(tests.get_nas_file('synthetic-persons-IL-2016.parquet'))).to_dataframe()
        self.df_b.name = 'df_b'

    @staticmethod
    def mean_calculation(column, df_a, df_b):
        df_a_column_mean = df_a[column].mean()
        df_b_column_mean = df_b[column].mean()
        return {'column': column, 'df_a_column_mean': df_a_column_mean,
                'df_b_column_mean': df_b_column_mean}

    @staticmethod
    def calculate_in_range(column, df_a_column_mean, df_b_column_mean, mean_range):
        return BooleanResult(value=abs(df_a_column_mean - df_b_column_mean) <= mean_range,
                             description=f'The mean of {column} is within the range {mean_range}')

    @staticmethod
    def calculate_percent(column, df_a_column_mean, df_b_column_mean):
        percentage_difference = abs(((df_a_column_mean - df_b_column_mean) / (df_a_column_mean)) * 100)
        return BooleanResult(value=percentage_difference,
                             description=f'The percent different of {column} is ')

    @staticmethod
    def count_unique(column, df_a):
        return {'unique_count': df_a[column].nunique()}

    @staticmethod
    def check_none(unique_count):
        return unique_count == 0

    def test_basic_analytic_class(self):
        test_analytic = Analytic()
        test_analytic.add_filter(query='INTP > 10000 and INTP < 100000')
        test_analytic.add_aggregate(aggregator=AnalyticTest.mean_calculation,
                                    comparator=AnalyticTest.calculate_in_range,
                                    extra_aggregator_params={'column': 'INTP'},
                                    extra_comparator_params={'mean_range': 500})
        test_analytic.add_aggregate(aggregator=AnalyticTest.mean_calculation,
                                    comparator=AnalyticTest.calculate_percent,
                                    extra_aggregator_params={'column': 'INTP'})
        filtered_df, aggregated_results = test_analytic.execute([self.df_a, self.df_b])
        print()

    def test_basic_consistency(self):
        test_consistency = Assertion()
        test_consistency.add_filter(query='INTP > 10000 and INTP < 100000')
        test_consistency.add_aggregate(aggregator=AnalyticTest.count_unique,
                                       comparator=AnalyticTest.check_none,
                                       extra_aggregator_params={'column': 'INTP'})
        filtered_df, aggregated_results = test_consistency.execute(df=self.df_a,
                                                                   default_df_name=False)
        print()
