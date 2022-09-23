import logging
from enum import Enum, unique
from typing import Union, Iterable
from numbers import Number

import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from censyn.metrics import Metrics
from ..results.result import PandasResult, PlotResult, Result


@unique
class ScalingStrategy(Enum):
    raw = 1
    percent_difference = 2
    weighted_percent_difference = 3
    group_weighted_percent_difference = 4


def raw_scaling(input1: Union[Number, Iterable[Number]], input2: Union[Number, Iterable[Number]]):
    """
    Computes the raw count differences between 2 datasets.

    :param input1: The original (reference) data
    :param input2: The new data to compare to reference
    :return: The difference between 2 datasets
    """
    return input2 - input1


def percent_difference_scaling(input1: Union[Number, Iterable[Number]], input2: Union[Number, Iterable[Number]]):
    """
    Computes the percent differences of each group in two datasets. E.g., 15 males vs 10 males == 0.5

    :param input1: The original (reference) data
    :param input2: The new data to compare to reference
    :return: The percent difference between two datasets
    """
    diffs = raw_scaling(input1, input2)
    diffs /= input1
    return diffs


def weighted_percent_difference(input1: Union[Number, Iterable[Number]], input2: Union[Number, Iterable[Number]],
                                weights: Iterable[Number]):
    """
    Computes percent difference of each group in two datasets weighted by given factor. Common choices might be:
    1) Whole population
    2) Size of group relative to other groups (e.g., young men and women would be weighted approx. equally,
       but older women weighted more than older men)

    :param input1: The original (reference) data
    :param input2: The new data to compare to reference
    :param weights: Weights for each entry in df1/2, used to scale the error
    :return: weighted percent difference between datasets
    """
    diffs = percent_difference_scaling(input1, input2)
    diffs *= weights
    return diffs


class TableMetrics(Metrics):
    def __init__(self, scaling_strategy: Union[ScalingStrategy, str] = ScalingStrategy.raw, pivot: bool = False,
                 file_name=None, *args, **kwargs):
        """
        TableMetrics can report group-by-group difference between real and synthetic datasets.

        :param name: Name of the metric
        :param scaling_strategy: How to scale errors between datasets
        :param pivot: Whether to output a pivot table of the given features or a flat list of differences
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate TableMetrics with scaling_strategy {scaling_strategy}, "
                          f"pivot {pivot} and file_name {file_name}.")

        if not self.features:
            raise ValueError('No features provided')

        if isinstance(scaling_strategy, str):
            self.scaling_strategy = ScalingStrategy[scaling_strategy]
        elif isinstance(scaling_strategy, ScalingStrategy):
            self.scaling_strategy = scaling_strategy
        else:
            raise ValueError('Invalid ScalingStrategy type')
        self.pivot = pivot
        self.file_name = file_name

    def table_metric_bar_plot(self, results: PandasResult, error_cutoff: float = 0.0, filepath: str = None, name=None,
                              scaling_strategy_name=None, features=None):
        """
        Takes the output of a TableMetrics group by diff and plots as a bar plot visualizing percent error
        between real and synthetic datasets

        :param features:
        :param scaling_strategy_name:
        :param name:
        :param results: pd.Series of percent errors
        :param error_cutoff: Minimum percent error you want plotted
        :param filepath: Optional. Where to save graph
        :return: an Axes object of the barplot of errors
        """
        name = name if name is not None else self.name
        scaling_strategy_name = scaling_strategy_name if scaling_strategy_name is not None else self.scaling_strategy.name
        features = features if features is not None else self.features
        assert error_cutoff >= 0, 'Error_cutoff must be a positive float'
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(16, 9))
        results = results.value.loc[results.value['abs_error'] >= error_cutoff]
        if len(results) == 0:
            raise ValueError('Error cutoff too high. No data to plot')
        results.plot(y='error', kind='bar', ax=ax)
        plt.title(f'{name} - {scaling_strategy_name}')
        plt.xlabel('(' + ', '.join(features) + ')')
        plt.ylabel('Error')
        if filepath:
            plt.savefig(filepath)
        return ax

    def table_metric_heatmap(self, results: PandasResult, max_error: float = 1.0, min_error: float = -1.0,
                             filepath: str = None, name=None, scaling_strategy_name=None):
        """
        Generate a heatmap representation of a pivot table of errors

        :param results: Output of TableMetrics pivot table
        :param max_error: max error for color gradient
        :param min_error: min error for color gradient
        :param filepath: Where to save figure
        :return: Axes object with heatmap of pivot table
        """
        name = name if name is not None else self.name
        scaling_strategy_name = scaling_strategy_name if scaling_strategy_name is not None else self.scaling_strategy.name
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(16, 9))
        # sns.heatmap(results.value, center=0, ax=ax, vmin=min_error, vmax=max_error)
        plt.title(f'{name} - {scaling_strategy_name}')
        if filepath:
            plt.savefig(filepath)
        return ax

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Computes output of TableMetric

        :param data_frame_a: First of data frames to evaluate
        :param data_frame_b: Second of data frames to evaluate
        :param weight_a: weight of the first of two data frames to evaluate
        :param weight_b: weight of the second of two data frames to evaluate
        :param add_a: Data frames for additional first data set
        :param add_b: Data frames for additional second data set
        :return: PandasResult
        """
        # check that data_frames share same features
        self.validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)
        self.data_frame_a = data_frame_a
        self.data_frame_b = data_frame_b

        function_dict = {
            "name": self.name,
            "scaling_strategy_name": self.scaling_strategy.name
        }
        if self.pivot:
            res = self._compute_pivot()
            to_return = PlotResult(value=res, function=self.table_metric_heatmap, function_dict=function_dict, file_name=self.file_name, metric_name=self.name) if self.file_name is not None else res
        else:
            res = self._compute_groupby_diffs()
            function_dict.update({'features': self.features})
            to_return = PlotResult(value=res, function=self.table_metric_bar_plot, function_dict=function_dict, file_name=self.file_name, metric_name=self.name) if self.file_name is not None else res
        return to_return

    def _scale(self, df1: pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
        """
        Applies the chosen ScalingStrategy to the computed differences

        :param df1: The original dataframe to compare
        :param df2: The second dataframe to compare
        :return: pd.DataFrame or pd.Series of differences between df2 and df1
        """
        if self.scaling_strategy == ScalingStrategy.raw:
            return raw_scaling(df1, df2)

        elif self.scaling_strategy == ScalingStrategy.percent_difference:
            return percent_difference_scaling(df1, df2)

        elif self.scaling_strategy == ScalingStrategy.weighted_percent_difference:
            population_weight = df1 / len(self.data_frame_a)
            return weighted_percent_difference(df1, df2, population_weight)

        elif self.scaling_strategy == ScalingStrategy.group_weighted_percent_difference:
            if len(df1.shape) == 1:
                weights = df1 / df1.max()

            elif len(df1.shape) == 2:
                weights = df1 / df1.max().max()

            else:
                raise ValueError("Only pandas Series and non-multi index DataFrames are supported as weights")

            return weighted_percent_difference(df1, df2, weights)

    def _compute_groupby_diffs(self) -> PandasResult:
        """
        Computes the difference in size of the 2 datasets grouped by the given features. The differences are scaled
        using the scaling strategy

        :return: PandasResult
        """
        gb1 = self.data_frame_a.groupby(self.features).size()
        gb2 = self.data_frame_b.groupby(self.features).size()

        diffs = self._scale(gb1, gb2)

        diffs = diffs.to_frame('error')
        diffs['abs_error'] = diffs['error'].abs()
        diffs = diffs.sort_values('abs_error', ascending=False)
        return PandasResult(diffs, metric_name=self.name)

    def _compute_pivot(self) -> PandasResult:
        """
        Creates a pivot table with a count in each cell. Currently, only supports 2 features (1 as index, 1 as columns).
        However, it theoretically could support more it'd just be ugly.

        :return: PandasResult
        """
        ct1 = pd.crosstab(index=self.data_frame_a[self.features[0]], columns=[self.data_frame_a[f] for f in self.features[1:]])
        ct2 = pd.crosstab(index=self.data_frame_b[self.features[0]], columns=[self.data_frame_b[f] for f in self.features[1:]])
        diffs = self._scale(ct1, ct2)
        return PandasResult(diffs, metric_name=self.name)
