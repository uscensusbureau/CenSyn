from typing import List, Dict

import pandas as pd

from censyn.metrics import Metrics
from censyn.results import Result


class Evaluator:
    """Handles evaluation of two dataframes according to Metrics to create a Report."""

    def __init__(self, df_1: pd.DataFrame, df_2: pd.DataFrame, evaluation_metrics: List[Metrics]) -> None:
        """
        Initialization of Evaluator.

        :param df_1: first pd.DataFrame
        :param df_2: second pd.DataFrame
        :param evaluation_metrics: list of Metrics used to evaluate df_1 and df_2
        """

        # Make sure there's something to work with.
        if df_1 is None or df_2 is None:
            raise ValueError('Attempted initialization of Evaluator with one or more pd.DataFrames of type None')

        # Do a header check.
        if list(df_1.columns.values) != list(df_2.columns.values):
            raise ValueError('Attempted instantiation of Evaluator with DataFrames that do not share the same header')

        # Error check the evaluation metrics.
        if evaluation_metrics is None or len(evaluation_metrics) == 0:
            raise ValueError('You must provide evaluation metrics to the Evaluator.')

        self._df_1: pd.DataFrame = df_1
        self._df_2: pd.DataFrame = df_2
        self._evaluation_metrics: List[Metrics] = evaluation_metrics
        self._evaluation_results: Dict[str, Result] = {}

    @property
    def evaluation_metrics(self) -> List[Metrics]:
        """Getter for evaluation metrics"""
        return self._evaluation_metrics

    @property
    def evaluation_results(self) -> Dict[str, Result]:
        """Getter for evaluation results"""
        return self._evaluation_results

    def evaluate(self, rerun=False) -> Dict[str, Result]:
        """
        Use the set of metrics in _evaluation_metrics to evaluate df_1 and df_2 and obtain metric results.

        :param rerun: This indicates if you want to rerun all of the metrics against the metrics.
        :return: A dictionary of results from metric name to a list of metric results.
        """
        if rerun or len(self.evaluation_results) == 0:
            self._evaluation_results = {metric.name: metric.compute_results(self._df_1, self._df_2,
                                                                            None, None, None, None)
                                        for metric in self.evaluation_metrics}
        return self.evaluation_results
