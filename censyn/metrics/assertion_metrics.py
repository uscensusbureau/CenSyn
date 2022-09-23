import logging
from typing import Union

import pandas as pd

from censyn.metrics import Metrics
from ..results.result import Result, ListResult
from censyn.asserts import ACS_asserts


class AssertionMetrics(Metrics):
    def __init__(self, name:str = 'AssertionMetrics',
                 #assertions: List[Assertion],
                 *args, **kwargs):
        """
        AssertionMetrics for evaluating and returning assertions

        :param data_frame_a: First of dataframes to evaluate
        :param data_frame_b: Second of dataframes to evaluate
        """
        super().__init__(name=name, *args, **kwargs)
        logging.debug(msg=f"Instantiate AssertionMetrics.")

        self._assertions = ACS_asserts.ALL_ACS_ASSERTS_LIST # TODO: pass this in as a parameter

    @property
    def assertions(self):
        """Getter for assertions"""
        return self._assertions

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Computes results from running assertions list on both dataframes

        :param data_frame_a: First of data frames to evaluate
        :param data_frame_b: Second of data frames to evaluate
        :param weight_a: weight of the first of two data frames to evaluate
        :param weight_b: weight of the second of two data frames to evaluate
        :param add_a: Data frames for additional first data set
        :param add_b: Data frames for additional second data set
        :return: ListResult[AssertionBooleanResult]
        """
        # check that data_frames share same features
        self.validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)
        self.data_frame_a = data_frame_a
        self.data_frame_b = data_frame_b

        # TODO: Hacky right now, this needs to be improved
        df_1 = [item.execute(df=self.data_frame_a)[1] for item in self.assertions]
        df_1 = [i for i in df_1[0]]
        for res in df_1:
            res.metric_name='DataFrame 1'
        df_2 = [item.execute(df=self.data_frame_b)[1] for item in self.assertions]
        df_2 = [i for i in df_2[0]]
        for res in df_2:
            res.metric_name = 'DataFrame 2'
        df_1_results = ListResult(value=df_1,
                                  metric_name='data_frame_a_assertions', description='Assertions over Dataframe A')
        df_2_results = ListResult(value=df_2,
                                  metric_name='data_frame_b_assertions', description='Assertions over Dataframe B')

        to_return = ListResult(value=[df_1_results, df_2_results], metric_name='list_metric',
                               description='All of the results for the marginal metric computation')
        return to_return
