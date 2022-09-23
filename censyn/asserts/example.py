from censyn import analytics
from typing import List

from censyn.results import AssertBooleanResult


class ExampleAssert:

    def __init__(self):
        self._assert_list = []
        self._dataset_asserts()

    @property
    def assert_list(self) -> List[analytics.Assertion]:
        return self._assert_list

    # The values in the returning need to match the variable name for your comparator.
    # This is because we need to know how to pass the information to the comparator
    @staticmethod
    def get_dataframe_details(df):
        return {'df': df.describe()}

    @staticmethod
    def mean_over_zero(df, column):
        return AssertBooleanResult(value=df[column]['mean'] > 0,
                                   expected_value=True,
                                   description='The mean is over zero')

    @staticmethod
    def max_over(df, column, max_value):
        return AssertBooleanResult(value=df[column]['max'] < max_value,
                                   expected_value=True,
                                   description=f'The {column} max value is less than {max_value}')

    def _dataset_asserts(self):
        self._assert_list = []
        check_1 = analytics.Assertion()
        check_1.add_filter(query='INTP > 10000 and INTP < 100000')
        check_1.add_aggregate(aggregator=ExampleAssert.get_dataframe_details,
                              comparator=ExampleAssert.mean_over_zero,
                              extra_comparator_params={'column': 'INTP'})
        self._assert_list.append(check_1)
        check_2 = analytics.Assertion()
        check_2.add_filter(query='INTP > 10000 and INTP < 100000')
        check_2.add_aggregate(aggregator=ExampleAssert.get_dataframe_details,
                              comparator=ExampleAssert.mean_over_zero,
                              extra_comparator_params={'column': 'INTP'})
        self._assert_list.append(check_2)
        check_3 = analytics.Assertion()
        check_3.add_aggregate(aggregator=ExampleAssert.get_dataframe_details,
                              comparator=ExampleAssert.max_over,
                              extra_comparator_params={'column': 'INTP', 'max_value': 1000000})
        self._assert_list.append(check_3)
