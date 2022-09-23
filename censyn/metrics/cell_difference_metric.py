import datetime
import logging

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from censyn.metrics import Metrics
from censyn.results.result import Result, ResultLevel, IntResult, FloatResult, ListResult, MappingResult
from censyn.results.table_result import TableResult


class CellDifferenceMetric(Metrics):

    def __init__(self, name: str = 'CellDifferenceMetric', align_features: List[str] = None, *args, **kwargs):
        """
        Initialization for CellDifferenceMetric

        :param name: Name for the metric
        :param align_features: List of features for aligning the data sets.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, name=name, **kwargs)
        logging.debug(msg=f"Instantiate CellDifferenceMetric with align_features {align_features}.")
        self._start_time = datetime.datetime.now()
        self._align_features = align_features

    @property
    def additional_features(self) -> List[str]:
        """Getter for additional features."""
        if self._align_features:
            return self._align_features
        return []

    def validate_data(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                      weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                      add_a: Union[pd.DataFrame, None], add_b: Union[pd.DataFrame, None]):
        """
        Check that the data sets are valid.

        :param data_frame_a: First of data frames to evaluate
        :param data_frame_b: Second of data frames to evaluate
        :param weight_a: weight of the first of two data frames to evaluate
        :param weight_b: weight of the second of two data frames to evaluate
        :param add_a: Data frames for additional first data set
        :param add_b: Data frames for additional second data set
        """
        super().validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

        # The data must be of equal sizes
        if data_frame_a.shape != data_frame_b.shape:
            if add_a is None or add_a.empty or add_b is None or add_b.empty:
                logging.error(msg=f"DataFrames are different shapes {data_frame_a.shape} vs. {data_frame_b.shape}")
                raise ValueError(f"DataFrames are different shapes {data_frame_a.shape} vs. {data_frame_b.shape}")

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Computes evaluation of the two data sets and create results.

        :param data_frame_a: First of data frames to evaluate
        :param data_frame_b: Second of data frames to evaluate
        :param weight_a: weight of the first of two data frames to evaluate
        :param weight_b: weight of the second of two data frames to evaluate
        :param add_a: Data frames for additional first data set
        :param add_b: Data frames for additional second data set
        :return: Result
        """
        self._start_time = datetime.datetime.now()
        self.validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

        if add_a is None or add_a.empty or add_b is None or add_b.empty:
            return self._compute(data_frame_a, data_frame_b, weight_a, weight_b)

        columns = [col for col in add_a.columns]
        logging.info(msg=f"Aligning data on {columns}")
        add_a['index_add_a'] = data_frame_a.index.values
        add_b['index_add_b'] = data_frame_b.index.values
        common_df = pd.merge(add_a, add_b, how='inner', on=columns, validate="one_to_one")
        mod_a_df = data_frame_a.loc[common_df['index_add_a'].tolist()].reset_index(drop=True, inplace=False)
        mod_b_df = data_frame_b.loc[common_df['index_add_b'].tolist()].reset_index(drop=True, inplace=False)
        if weight_a is not None and not weight_a.empty and weight_b is not None and not weight_b.empty:
            mod_weight_a = weight_a.loc[common_df['index_add_a'].tolist()].reset_index(drop=True, inplace=False)
            mod_weight_b = weight_b.loc[common_df['index_add_b'].tolist()].reset_index(drop=True, inplace=False)
        else:
            mod_weight_a = None
            mod_weight_b = None

        self.validate_data(mod_a_df, mod_b_df, mod_weight_a, mod_weight_b, None, None)
        res = self._compute(mod_a_df, mod_b_df, mod_weight_a, mod_weight_b)
        if isinstance(res, ListResult):
            for cur_r in res.value:
                if cur_r.description == "Summary total":
                    if mod_a_df.shape[0] < data_frame_a.shape[0] or mod_b_df.shape[0] < data_frame_b.shape[0]:
                        cur_r.value["Data A not analyzed "] = f"{data_frame_a.shape[0] - mod_a_df.shape[0]} rows."
                        cur_r.value["Data B not analyzed "] = f"{data_frame_b.shape[0] - mod_b_df.shape[0]} rows."

        return res

    def _compute(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                 weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None]) -> Result:
        """
        Computes evaluation of the two data sets and create results.

        :param data_frame_a: First of data frames to evaluate
        :param data_frame_b: Second of data frames to evaluate
        :param weight_a: weight of the first of two data frames to evaluate
        :param weight_b: weight of the second of two data frames to evaluate
        :return: Result
        """
        def median_difference(feature_name: str, feat_diff_indices) -> Tuple[Any, int]:
            """
            Generate the median for the difference of the feature's data.

            :param feature_name: String of feature name
            :param feat_diff_indices: Indices of the differences for the feature.
            :return: Numeric median value, Count of the NA
            """
            a_df, b_df = data_frame_a.iloc[feat_diff_indices], data_frame_b.iloc[feat_diff_indices]
            if not np.issubdtype(a_df[feature_name], np.number) or not np.issubdtype(b_df[feature_name], np.number):
                return np.nan, a_df[feature_name].isna().sum() + b_df[feature_name].isna().sum()
            diffs = abs(a_df[feature_name] - b_df[feature_name])
            if diff_weight is not None:
                weight_df = pd.concat([diffs, weight_s.iloc[feat_diff_indices]], axis=1)
                weight_df.columns = ['feat', 'wt']
                weight_df.dropna(axis=0, inplace=True)
                if weight_df.empty:
                    return 0, diffs.size
                weight_df.sort_values(by=['feat'], inplace=True)
                per_50 = weight_df['wt'].sum() / 2.0
                cum_sum = weight_df['wt'].cumsum()
                w_median = weight_df['feat'][cum_sum >= per_50].iloc[0]
                return w_median, diffs.size - weight_df.shape[0]
            na_count = diffs.isna().sum()
            med = np.nan if na_count == diffs.shape[0] else diffs.median(axis=0, skipna=True)
            return med, na_count

        def feature_diff(feature_name: str, feat_df: pd.DataFrame) -> Dict:
            """
            Generate the difference of the feature's data.

            :param feature_name: String of feature name
            :param feat_df: DataFrame of the feature differences.
            :return: Dictionary of the difference statistics for the feature
            """
            if feat_df.empty:
                return {"Feature": feature_name, "Count": 0, "NA Count": 0, "Median Difference": 0,
                        "Percent Changed": 0}
            median, na_count = median_difference(col, feat_df.index.get_level_values('id'))
            return {"Feature": feature_name,
                    "Count": feat_df.shape[0],
                    "NA Count": na_count,
                    "Median Difference": median,
                    "Percent Changed": feat_df.shape[0] * 100 / total_rows
                    }

        # Generate difference between data sets
        logging.info(f'Data size {data_frame_a.shape[0]} rows by  {data_frame_a.shape[1]} columns')
        diff_df = self.diff_pd(df1=data_frame_a, df2=data_frame_b)
        total_rows = len(data_frame_a.index)

        # Check if no difference
        if diff_df is None:
            res = IntResult(value=0)
            return res
        # Set difference weight values
        if weight_a is not None and weight_b is not None:
            weight_s = weight_a + weight_b
            diff_weight = weight_s.iloc[diff_df.index.get_level_values('id')]
        else:
            weight_s = None
            diff_weight = None

        total_differences = {
            "Analyze Data Size": f"{data_frame_a.shape[0]} rows by {data_frame_a.shape[1]} columns.",
            "Total Differences": diff_df.shape[0],
            "Total Differences %": (diff_df.shape[0] / (data_frame_a.shape[0] * data_frame_a.shape[1])) * 100,
            "Average number of differences per row": diff_df.shape[0] / data_frame_a.shape[0]
        }
        if diff_weight is not None:
            total_sum = np.sum(weight_s) * data_frame_a.shape[1]
            total_differences["Weighted Total Differences %"] = diff_weight.sum() * 100 / total_sum
        summary_r = MappingResult(value=total_differences, metric_name=self.name,
                                  description="Summary total", level=ResultLevel.SUMMARY)
        summary_r.display_number_lines = 0

        diff_feats = {}
        columns = diff_df.index.get_level_values('col').unique()
        for col in columns:
            col_df = diff_df[np.in1d(diff_df.index.get_level_values('col'), [col])]
            diff_feats[col] = feature_diff(feature_name=col, feat_df=col_df)
        feature_data = pd.DataFrame(diff_feats).T
        individual_r = TableResult(value=feature_data, metric_name=self.name, sort_column="Count", ascending=False,
                                   description="Individual feature statistics", level=ResultLevel.SUMMARY)
        individual_r.display_int_auto = ["Median Difference"]
        individual_r.display_float_auto = ["Percent Changed"]

        time_r = FloatResult(value=datetime.datetime.now() - self._start_time, metric_name=self.name,
                             description="Time", level=ResultLevel.SUMMARY)

        to_return = ListResult(value=[time_r, summary_r, individual_r], metric_name=self.name,
                               description="All results for the cell difference metric")
        return to_return

    @staticmethod
    def diff_pd(df1: pd.DataFrame, df2: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Identify differences between two pandas DataFrames.

        :param df1: First of data frames to evaluate
        :param df2: Second of data frames to evaluate
        :return: Result
        """
        assert (df1.columns == df2.columns).all(), \
            "DataFrame column names are different"
        if any(df1.dtypes != df2.dtypes):
            "Data Types are different, trying to convert"
            for i in range(df1.shape[1]):
                type_1 = df1[df1.columns[i]].dtype
                type_2 = df2[df2.columns[i]].dtype
                if type_1 != type_2:
                    if type_1 == 'float':
                        df2[df2.columns[i]].astype(df1[df1.columns[i]].dtype)
                    elif type_2 == 'float':
                        df1[df1.columns[i]].astype(df2[df1.columns[i]].dtype)
                    else:
                        df2[df2.columns[i]].astype(df2[df1.columns[i]].dtype)
        if df1.equals(df2):
            return None
        else:
            # need to account for np.nan != np.nan returning True
            diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
            ne_stacked = diff_mask.stack()
            changed = ne_stacked[ne_stacked]
            changed.index.names = ['id', 'col']
            difference_locations = np.where(diff_mask)
            changed_from = df1.values[difference_locations]
            changed_to = df2.values[difference_locations]
            return pd.DataFrame({'from': changed_from, 'to': changed_to},
                                index=changed.index)
