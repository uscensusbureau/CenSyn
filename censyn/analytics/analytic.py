import copy
import operator
from functools import partial
from typing import List

import numpy as np
import pandas as pd


class Aggregate:
    def __init__(self, aggregator, comparator, extra_aggregator_params=None, extra_comparator_params=None):
        self.aggregator = aggregator
        self.extra_aggregator_params = extra_aggregator_params or {}
        self.comparator = comparator
        self.extra_comparator_params = extra_comparator_params or {}


class Analytic:
    def __init__(self):
        self.__aggregates = []
        self._test_filter = []

    @property
    def aggregates(self):
        """Returns all of the aggregate functions"""
        return self.__aggregates

    def add_filter(self, query):
        """
        This adds a query to be run on any set of dataframes at a later time.

        :param query: This is a pandas query to be run on a set of dataframes at a later
        time. This follows the standard way that pandas queries are written.
        """
        self._test_filter.append(query)
        return self

    def add_aggregate(self, aggregator, comparator=None, extra_aggregator_params=None,
                      extra_comparator_params=None):
        """
        This adds a function to be run when execute is called on this class.

        :param function: This is one of the functions that will be used to aggregate the
        filtered data.
        :param  extra_function_params: The standard function that gets sent into this
        must have parameters for each dataframe. However, you want to have some other
        paramters sent to the function when it is called you need to provide a dict that
        maps from paramters name to value you want to be sent in.
        :return this returns self to support chain calling the function.
        """
        self.aggregates.append(Aggregate(aggregator=aggregator,
                                         comparator=comparator,
                                         extra_aggregator_params=extra_aggregator_params,
                                         extra_comparator_params=extra_comparator_params))
        return self

    def _run_filter(self, dataframes):
        all_dfs = {df.name: df for df in dataframes}
        filtered_dataframes = {}
        if self._test_filter:
            for key, df in all_dfs.items():
                df_to_apply = df.copy(deep=True)
                for value in self._test_filter:
                    df_to_apply = df_to_apply.query(value)
                filtered_dataframes[key] = df_to_apply
        else:
            filtered_dataframes = copy.deepcopy(all_dfs)
        return filtered_dataframes

    def _run_aggregates(self, filtered_dataframes):
        aggregate_results = []
        for aggregate in self.aggregates:
            result = aggregate.aggregator(**filtered_dataframes,
                                          **aggregate.extra_aggregator_params)
            if aggregate.comparator:
                result = aggregate.comparator(**result, **aggregate.extra_comparator_params)
            aggregate_results.append(result)
        return aggregate_results

    def execute(self, dataframes):
        """
        Applies the masks that where created with add_filter and then either returns back
        the filtered dataframes or the result of the functions you passed in with
        add_aggregate.

        :param dataframes: All the dataframes you want to execute this Analytic on.
        :return: This returns a tuple the first element is the filtered dataframes and the
        second element is a list of the results of the aggregations.
        """
        filtered_dataframes = self._run_filter(dataframes=dataframes)
        aggregate_results = self._run_aggregates(filtered_dataframes=filtered_dataframes)

        return filtered_dataframes, aggregate_results


class Assertion(Analytic):
    """
    A class that extends Analytic. It is just a easy way for a user to execute the analytic
    class verses a single dataframe.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_aggregate(self, aggregator, comparator=None, extra_aggregator_params=None,
                      extra_comparator_params=None):
        if comparator is None:
            raise ValueError('You must provide a comparator.')
        super().add_aggregate(aggregator, comparator, extra_aggregator_params,
                              extra_comparator_params)

    def execute(self, df, default_df_name: bool=True):
        """
        If you provided and filter it applies them it also run any aggregates that where
        provided.

        :param default_df_name: If this is true we set a default name for the dataframe.
        :return: This returns a tuple the first element is the filtered dataframes and the
        second element is a list of the results of the aggregations.
        """
        if default_df_name:
            df.name = 'df'
        return super().execute([df])


class DifferentialCheck(Analytic):
    """
    A class that extends Analytic. It is just a easy way for a user to execute the analytic
    class verses two dataframes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, df_a, df_b, default_df_name: bool=True):
        """
        If you provided and filter it applies them it also run any aggregates that where
        provided.

        :param default_df_name: If this is true we set a default name for the dataframe.
        :return: This returns a tuple the first element is the filtered dataframes and the
        second element is a list of the results of the aggregations.
        """
        if default_df_name:
            df_a.name = 'df_a'
            df_b.name = 'df_b'
        return super().execute([df_a, df_b])
