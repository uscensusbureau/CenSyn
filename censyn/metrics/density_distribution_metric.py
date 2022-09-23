import logging
from typing import Dict, List, Union

import pandas as pd

from censyn.metrics import MarginalMetric, PickingStrategy
from censyn.results import result, ResultLevel, StableFeatureResult
from censyn.results.result import Result, ListResult
from censyn.utils import compute_density_total_difference, StableFeaturesColumn, StableFeatures


class DensityDistributionMetric(MarginalMetric):

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for DistortionMetric

        :param: args:
        :param: kwargs:
        """
        super().__init__(*args, marginal_dimensionality=1, picking_strategy=PickingStrategy.rolling,
                         sample_ratio=1.0, *args, **kwargs)
        logging.debug(msg=f"Instantiate DensityDistributionMetric.")
        self._data_a_count = 0
        self._data_b_count = 0

    def validate_data(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                      weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                      add_a: Union[pd.DataFrame, None], add_b: Union[pd.DataFrame, None]):
        """
        Check that the data sets are valid.

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        """
        super().validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Calculate density distribution results of the data.

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        :return: one Result obj, containing all component Result objs
        """
        self.validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)
        self._data_a_count = data_frame_a.shape[0]
        self._data_b_count = data_frame_b.shape[0]

        cur_results = super().compute_results(data_frame_a, data_frame_b, weight_a=weight_a, weight_b=weight_b,
                                              add_a=add_a, add_b=add_b)

        ret_results = self._update_results(cur_results)
        return ret_results

    def _update_results(self, in_result: Result) -> ListResult:
        """
        Create the metrics results.

        :param: in_result: The base marginal metrics results.
        :return: a ListResult
        """
        def _process_result(cur_result: Result, results: Dict) -> None:
            """
            Process the current Result to the results.

            :param: cur_result: Results
            :param: results: Dictionary of current output results.
            :return: None
            """
            if cur_result.container:
                if isinstance(cur_result.value, List):
                    for res in iter(cur_result.value):
                        if res is not None:
                            _process_result(res, results)
                elif isinstance(cur_result.value, Dict):
                    for res in iter(cur_result.value):
                        if res is not None:
                            _process_result(res, results)
                else:
                    ValueError("invalid result.")
            elif cur_result.description.startswith("Summary:\n"):
                summary_result = result.MappingResult(value=cur_result.value.copy(), metric_name=self.name,
                                                      description=f"summary of results",
                                                      level=ResultLevel.SUMMARY)
                summary_result.display_number_lines = 0
                results["summary"] = summary_result
            elif cur_result.description.startswith("Individual scores"):
                max_len = max([len(feat[0][0]) for feat in cur_result.value])
                individual_scores = [f"{feat[0][0].ljust(max_len, ' ')}    {feat[1]}" for feat in cur_result.value]
                individual_result = result.ListResult(value=individual_scores, metric_name=self.name,
                                                      description=f"Feature error scores",
                                                      level=ResultLevel.SUMMARY)
                individual_result.display_number_lines = 0
                results["individual"] = individual_result
            elif cur_result.description.startswith("Per-value-tuple"):
                feature_results = []
                for feat, dist in cur_result.value.items():
                    feature_name = feat[0]
                    cur_feature_results = []
                    mean_density_dist_diffs = compute_density_total_difference(dist)
                    k_marginal_score = {
                        'Raw Error Score': mean_density_dist_diffs,
                        'Calculated NIST Score': self._scale_raw_marginal_score(mean_density_dist_diffs)
                    }
                    score_result = result.MappingResult(value=k_marginal_score, metric_name=self.name,
                                                        description=f"Score for {feature_name}",
                                                        level=ResultLevel.SUMMARY)
                    score_result.display_number_lines = 0
                    cur_feature_results.append(score_result)

                    sf = StableFeatures(names=feature_name, sort_col=StableFeaturesColumn.bins)
                    sf.add_scores(dist)
                    sf_result = StableFeatureResult(value=sf.scores(), metric_name=self.name,
                                                    level=ResultLevel.SUMMARY,
                                                    description=f"Density Distribution for {feature_name}",
                                                    sf=feat,
                                                    data_a_count=self._data_a_count,
                                                    data_b_count=self._data_b_count,
                                                    baseline_count=0)
                    sf_result.title = f"{feature_name} Density Distribution Values"
                    cur_feature_results.append(sf_result)
                    feature_results.append(ListResult(value=cur_feature_results, metric_name="DistributionMetric",
                                                      description=f"All of the density distribution results for the "
                                                                  f"distribution marginal metric computation"))
                results["features"] = ListResult(value=feature_results, metric_name="DistributionMetric",
                                                 description=f"All of the density distribution results for the "
                                                             f"distribution marginal metric computation")

        results_d = {}
        _process_result(in_result, results_d)
        results_l = [results_d["summary"], results_d["individual"], results_d["features"]]
        to_return = ListResult(value=results_l, metric_name="DistributionMetric",
                               description='All of the results for the distribution marginal metric computation')
        return to_return
