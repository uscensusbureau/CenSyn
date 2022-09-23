import datetime
import logging
import math
import random
from abc import ABC, abstractmethod
from enum import Enum, unique
from functools import partial
from itertools import combinations
from typing import Dict, List, Tuple, Union

import pandas as pd

from ..results import result, Result, ResultLevel, TableResult, StableFeatureResult
from censyn.utils import stable_features_decorator, compute_feature_combinations_to_density_distributions, \
    bounded_pool, frequent_item_set, \
    compute_density_distribution_differences, StableFeatures, compute_density_mean, \
    compute_mapping_feature_combinations_to_density_distribution_differences

MAXIMUM_COMBINATIONS = 1 * 2**23


@unique
class PickingStrategy(Enum):
    """
    Enum for strategy of selecting sets of features

    all: all possible combinations, for a given combination set size
    rolling: incrementally shifting sets. example: (1, 2, 3), (2, 3, 4), ... (8, 9, n)
    random: random selection from all possible combinations
    """
    lexicographic = 1  # lexicographic ordering
    rolling = 2
    random = 3


class Metrics(ABC):
    """Abstract class for metrics used to evaluate two dataframes."""

    def __init__(self, features: Dict = None, use_bins: bool = False, use_weights: bool = True,
                 logging_level=logging.INFO, name: str = None) -> None:
        """
        Initialization for base Metrics

        :param: features: Dictionary of the features.
        :param: use_bins: Boolean flag for the metric uses binned values of the data.
        :param: use_weights: Boolean flag for the metric to use weights for the data.
        :param: logging_level: The Logging level.
        :param: name: The name of the metric.
        """
        logging.debug(msg=f"Instantiate Metrics with features {features}, use_bins {use_bins}, "
                          f"use_weights {use_weights} and name {name}.")
        self._data_frame_a = None
        self._data_frame_b = None
        self._features = features
        self._use_bins = use_bins
        self._use_weights = use_weights

        if name is None:
            logging.error("Name must be specified for Metrics")
            raise ValueError("Name must be specified for Metrics")
        self._name: str = name

    @property
    def data_frame_a(self) -> pd.DataFrame:
        """Getter for data set a."""
        return self._data_frame_a

    @data_frame_a.setter
    def data_frame_a(self, value: pd.DataFrame):
        """Setter for data set a."""
        self._data_frame_a = value

    @property
    def data_frame_b(self) -> pd.DataFrame:
        """Getter for data set b."""
        return self._data_frame_b

    @data_frame_b.setter
    def data_frame_b(self, value: pd.DataFrame):
        """Setter for data set b."""
        self._data_frame_b = value

    @property
    def features(self) -> Dict:
        """Getter for the dictionary of features."""
        return self._features

    @property
    def use_bins(self) -> bool:
        """Getter for the use bins flag."""
        return self._use_bins

    @property
    def use_weights(self) -> bool:
        """Getter for the use weights flag."""
        return self._use_weights

    @property
    def name(self) -> str:
        return self._name

    @property
    def additional_features(self) -> List[str]:
        """Getter for additional features."""
        return []

    @property
    def in_process_features(self) -> List[str]:
        """Getter for in process features."""
        return []

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
        logging.debug(msg=f"Metrics.validate_data(). data_frame_a {data_frame_a.shape}, "
                          f"data_frame_b {data_frame_b.shape}.")
        if not data_frame_a.columns.equals(data_frame_b.columns):
            msg = f"Difference in columns of data frames used to initialize Metric. " \
                  f"Columns {data_frame_a.columns} is not equal to {data_frame_b.columns}"
            logging.error(msg=msg)
            raise ValueError(msg)
        if weight_a is not None:
            if weight_b is None:
                logging.error("Weight data set a exists but not for data set b.")
                raise ValueError("Weight data set a exists but not for data set b.")
            if data_frame_a.shape[0] != weight_a.shape[0]:
                msg = f"Data Set a size {data_frame_a.shape[0]} does not equal weight size {weight_a.shape[0]}"
                logging.error(msg=msg)
                raise ValueError(msg)
            if data_frame_b.shape[0] != weight_b.shape[0]:
                msg = f"Data Set b size {data_frame_a.shape[0]} does not equal weight size {weight_a.shape[0]}"
                logging.error(msg=msg)
                raise ValueError(msg)
        elif weight_b is not None:
            logging.error(msg="Weight data set b exists but not for data set a.")
            raise ValueError("Weight data set b exists but not for data set a.")
        if add_a is not None:
            if add_b is None:
                logging.error(msg="Additional data set a exists but not additional data set b.")
                raise ValueError("Additional data set a exists but not additional data set b.")
            if not add_a.columns.equals(add_b.columns):
                msg = f"Difference in columns of additional data frames. " \
                      f"Additional data a columns {add_a.columns} is not equal to additional data b {add_b.columns}"
                logging.error(msg=msg)
                raise ValueError(msg)
            if data_frame_a.shape[0] != add_a.shape[0]:
                msg = f"Data Set a {data_frame_a.shape[0]} does not equal additional data set {add_a.shape[0]}"
                logging.error(msg=msg)
                raise ValueError(msg)
            if data_frame_b.shape[0] != add_b.shape[0]:
                msg = f"Data Set b {data_frame_b.shape[0]} does not equal additional data set {add_b.shape[0]}"
                logging.error(msg=msg)
                raise ValueError(msg)
        elif add_b is not None:
            logging.error(msg="Additional data set b exists but not additional data set a.")
            raise ValueError("Additional data set b exists but not additional data set a.")

    def get_feature_name(self, col: str) -> str:
        """
        Get the feature name associated with a column name.

        :param: col: Name of the column data.
        :return: Feature name
        """
        if self.features:
            if col in self.features:
                return col
        msg = f"No feature associated with column {col}."
        logging.error(msg=msg)
        raise ValueError(msg)

    @abstractmethod
    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None):
        """
        Computes evaluation of the two data sets and create results.

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        :return: Result
        """
        logging.error("Attempted call to abstract method Metrics.compute_results()")
        raise NotImplementedError('Attempted call to abstract method Metrics.compute_results()')


class MarginalMetric(Metrics):
    """Metric for evaluating the similarity of two data sets according to a selection of n-marginals."""

    def __init__(self, name: str = 'MarginalMetric',
                 marginal_dimensionality: int = 3,
                 picking_strategy: PickingStrategy = PickingStrategy.lexicographic,
                 sample_ratio: float = 1.0, maximum_marginals: int = 0, baseline: float = 0.0,
                 cores: int = 4, stable_features: List[str] = None, min_support_item_set: float = 0.05,
                 item_set_percentage: float = 0.2, *args, **kwargs) -> None:
        """
        Initialization for MarginalMetric

        :param: marginal_dimensionality: set size for combinations to be selecting from
        :param: picking_strategy: how feature combinations are selected
        :param: sample_ratio: proportion of relevant feature combinations to select.
               The set of relevant combinations is determined by picking_strategy.
        :param: maximum_marginals: Maximum number of marginals to utilize. A value of 0 means unlimited.
        :param: Baseline percent.
        :param: stable_features: a list of features that stay stable, such that if you give feature 'A'
               it will give you k-marginals that include 'A'
        :param: min_support_item_set: at what point to cut off frequent item set mining.
               The support is the proportion of transactions in the dataset which contains the itemset.
        :param: item_set_percentage: percentage of the k marginal combinations you want to run frequent
               itemset mining on.
        """
        super().__init__(*args, name=name, **kwargs)
        logging.debug(msg=f"Instantiate MarginalMetric with marginal_dimensionality {marginal_dimensionality}, "
                          f"picking_strategy {picking_strategy}, sample_ratio {sample_ratio}, "
                          f"maximum_marginals {maximum_marginals}, baseline {baseline},"
                          f"stable_features {stable_features}, min_support_item_set {min_support_item_set}, "
                          f"item_set_percentage {item_set_percentage} and name {name}.")

        if marginal_dimensionality < 1:
            msg = f"Marginal_dimensionality of {marginal_dimensionality} passed in initialization of " \
                  f" MarginalMetric that is not greater than 0."
            logging.error(msg=msg)
            raise ValueError(msg)

        self._cores: int = cores
        self._item_set_percentage = item_set_percentage
        self._marginal_dimensionality = marginal_dimensionality
        self._min_support_item_set = min_support_item_set
        self._picking_strategy = picking_strategy

        if isinstance(picking_strategy, str):
            self._picking_strategy = PickingStrategy[picking_strategy]
        elif isinstance(picking_strategy, PickingStrategy):
            self._picking_strategy = picking_strategy
        else:
            msg = f"Metric {name} has invalid PickingStrategy type {picking_strategy}"
            logging.error(msg=msg)
            raise ValueError(msg)

        self._sample_ratio = sample_ratio
        self._maximum_marginals = maximum_marginals
        self._stable_features = stable_features
        self._baseline = baseline

    @property
    def in_process_features(self) -> List[str]:
        """Getter for in process features."""
        return self._stable_features if self._stable_features else []

    @property
    def item_set_percentage(self) -> float:
        return self._item_set_percentage

    @property
    def min_support_item_set(self) -> float:
        return self._min_support_item_set

    @property
    def marginal_dimensionality(self) -> int:
        """Getter for marginal_dimensionality"""
        return self._marginal_dimensionality

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
        logging.debug(msg=f"MarginalMetric.validate_data(). data_frame_a {data_frame_a.shape}, "
                          f"data_frame_b {data_frame_b.shape}.")
        super().validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

        if self._stable_features and not all((f_name in data_frame_a.columns for f_name in self._stable_features)):
            logging.error(msg="Stable features have to exists in the data.")
            raise ValueError("Stable features have to exists in the data.")

    def _number_marginals(self, number_combinations: int) -> int:
        """
        Calculate the number of marginals to utilize. It uses the properties sample ratio and maximum marginals.

        :param: number_combinations: Number of unique marginals.
        :return: number of marginals.
        """
        logging.debug(msg=f"MarginalMetric._number_marginals(). number_combinations {number_combinations}.")
        number_to_return = math.ceil(self._sample_ratio * number_combinations)
        if self._maximum_marginals > 0:
            if self._maximum_marginals < number_to_return:
                number_to_return = self._maximum_marginals
        return number_to_return

    @stable_features_decorator
    def _lexicographic_feature_combinations(self, features) -> List[tuple]:
        """
        Compute and return all feature combinations, for a specified marginal dimensionality.

        :param: features: This is a list of features/columns you want to use to make your combinations.
        :return: Returns list of all feature combinations of size == self.marginal_dimensionality
        """
        logging.debug(msg=f"MarginalMetric._lexicographic_feature_combinations().")
        all_combinations = list(combinations(features, self._marginal_dimensionality))
        number_to_return = self._number_marginals(len(all_combinations))
        all_combinations_to_return = all_combinations[:number_to_return]
        return all_combinations_to_return

    @stable_features_decorator
    def _random_feature_combinations(self, features) -> List[tuple]:
        """
        Compute and return a random set of feature combinations, for a specified marginal dimensionality,
        and sample ratio.

        :param: features: This is a list of features/columns you want to use to make your combinations.
        :return: A list of randomly selected feature combinations.
        """
        logging.debug(msg=f"MarginalMetric._random_feature_combinations().")

        def _number_combinations(n: int, r: int) -> int:
            return int(math.factorial(n) / (math.factorial(r) * math.factorial(n - r)))

        def _random_combinations(f, dimensionality) -> List[tuple]:
            random_combinations_set = []
            number_samples = math.ceil(_number_combinations(n=len(f), r=dimensionality) * ratio)
            if dimensionality > 1:
                while _number_combinations(n=len(f), r=dimensionality) > MAXIMUM_COMBINATIONS:
                    cur_feature = (f.pop(),)
                    cur_combos = _random_combinations(f, dimensionality - 1)
                    combos = [c + cur_feature for c in cur_combos]
                    random_combinations_set.extend(combos)
            if random_combinations_set:
                rest_samples = math.ceil(_number_combinations(n=len(f), r=dimensionality) * ratio)
                current_samples = number_samples - rest_samples
                random_combinations_set = random.sample(random_combinations_set, k=current_samples)
                random_combinations_set.extend(random.sample(list(combinations(f, dimensionality)), k=rest_samples))
            else:
                random_combinations_set = random.sample(list(combinations(f, dimensionality)), k=number_samples)
            return random_combinations_set

        num_total_combos = _number_combinations(n=len(features), r=self._marginal_dimensionality)
        num_combos = self._number_marginals(num_total_combos)
        ratio = max(num_combos / num_total_combos, 2**5 / MAXIMUM_COMBINATIONS)
        f_combos = _random_combinations(features.copy(), self._marginal_dimensionality)
        if len(f_combos) > num_combos:
            f_combos = random.sample(f_combos, num_combos)
        return f_combos

    @stable_features_decorator
    def _rolling_feature_combinations(self, features) -> List[tuple]:
        """
        Compute and return a set of rolling feature combinations.

        :param: features: This is a list of features/columns you want to use to make your combinations.
        :return: A list of incrementally shifting feature combinations. Example: (1, 2, 3), (2, 3, 4), ... (8, 9, n).
        """
        logging.debug(msg=f"MarginalMetric._rolling_feature_combinations().")
        set_size = self._marginal_dimensionality
        all_rolling_combinations = [tuple(features[i: i + set_size]) for i in range(len(features) - set_size + 1)]
        number_to_return = self._number_marginals(len(all_rolling_combinations))
        rolling_combinations_to_return = all_rolling_combinations[:number_to_return]
        return rolling_combinations_to_return

    def get_feature_combinations(self) -> List:
        """
        Get list of all relevant n-feature combinations, according to picking strategy property.
        It will not include the weight feature.

        :return: a list of feature combinations, selected according to the picking_strategy and sample_ratio
        """
        logging.debug(msg=f"MarginalMetric.get_feature_combinations().")
        features = list(self.data_frame_a.columns)
        if self._picking_strategy == PickingStrategy.lexicographic:
            return self._lexicographic_feature_combinations(features=features)

        if self._picking_strategy == PickingStrategy.random:
            return self._random_feature_combinations(features=features)

        if self._picking_strategy == PickingStrategy.rolling:
            return self._rolling_feature_combinations(features=features)

    def compute_mapping_feature_combinations_to_predicates(self) -> Tuple[Dict, Dict]:
        """
        For both data sets, return a mapping of {n-feature combinations : predicates}, where a predicate
        is a set of feature-values.

        :return: a dictionary of feature combinations to predicates
        """
        logging.debug(msg=f"MarginalMetric.compute_mapping_feature_combinations_to_predicates().")

        feature_combinations = self.get_feature_combinations()

        feature_combinations_to_predicates_a = {}
        feature_combinations_to_predicates_b = {}

        for feature_combination in feature_combinations:
            # get list of all observed predicates for both data frames
            observed_predicates_a = list(self.data_frame_a.groupby(by=list(feature_combination)).size().index)
            observed_predicates_b = list(self.data_frame_b.groupby(by=list(feature_combination)).size().index)

            feature_combinations_to_predicates_a[str(feature_combination)] = observed_predicates_a
            feature_combinations_to_predicates_b[str(feature_combination)] = observed_predicates_b

        return feature_combinations_to_predicates_a, feature_combinations_to_predicates_b

    def compute_combinations_density_distributions(self, feature_combinations: List,
                                                   data_frame: pd.DataFrame,
                                                   weight: Union[pd.Series, None]) -> Dict:
        """
        Compute the {n-feature-combos : {predicates : density}} for each data set and return a mapping of
        {n-feature-combos : density_distribution_differences} between the data sets, where differences are absolute.

        This density distribution is a data frame with each predicate as its index, so a mapping between predicates
        and density is still achieved.

        :return: A Dictionary for both DataFrames, that map each feature combination to a density_distributions.
                 Each Dictionary is of the form {n-feature-combo : [density distribution]}
        """
        logging.info(msg=f"MarginalMetric.compute_combinations_density_distributions(data_frame={data_frame.shape}).")

        # NOTE: GitLab Issue #33 specified a return type of:
        #       {n-feature-combos : {predicates : density}}, one for both data frame.
        #       Current return type however is:
        #       {n-feature-combo : [density distribution]}, one for both data frame.

        feature_combinations_to_density_distribution = {}

        # for each feature combination in a given set of feature combinations, compute density distribution
        result_gen = (
            partial(compute_feature_combinations_to_density_distributions,
                    feature_combination=feature_combination,
                    in_df=data_frame[list(feature_combination)],
                    in_weight=weight)
            for feature_combination in feature_combinations
        )
        results = bounded_pool(process_num=self._cores, functions=result_gen)

        # Each result is two tuples -- one each for df_a and df_b
        for tuple_res in results:
            feature_combinations_to_density_distribution[tuple_res[0]] = tuple_res[1]

        return feature_combinations_to_density_distribution

    @staticmethod
    def _scale_raw_marginal_score(mean_density_distribution_differences: float) -> float:
        """
        This is a function that scales the raw marginal score to a score that ranges from
        0 to 1000 where 1000 is a perfect score and 0 is the worst.

        :param: mean_density_distribution_differences the raw marginal score
        :return: A single number scaled between 0 and 1000
        """
        return ((2 - mean_density_distribution_differences) / 2) * 1000

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Instantiate 6 Results objects, of the following Result types:

        1. (Result, list) that contains Density Distributions, for both datasets
            1. a (Result, dict) - mapping of feature combos to density distributions for data set a
            1. b (Result, dict) - mapping of feature combos to density distributions for data set b
        2. (Result, list) Abs_Dif( Density Distributions), of both data sets
        3. (Result, float) Mean( Sum( Abs_Dif( Density Distributions) ) ), of both data sets
        4. (Result, list) containing all the above Result objs

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        :return: one Result obj, containing all component Result objs
        """
        logging.debug(msg=f"MarginalMetric.compute_results().")
        start_time = datetime.datetime.now()
        baseline_data_frame = None
        baseline_weight = None

        # check that data_frames share same features
        MarginalMetric.validate_data(self, data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)
        self.data_frame_a = data_frame_a
        self.data_frame_b = data_frame_b
        if self._baseline > 0.0:
            baseline_data_frame = data_frame_a.sample(frac=self._baseline / 100)
            if weight_a is not None:
                baseline_weight = weight_a.loc[baseline_data_frame.index]

        # Generate the metric name
        metric_name = f'{str(self._marginal_dimensionality)}-marginal'
        if self._stable_features:
            if len(self._stable_features) == 1:
                metric_name = f"{metric_name} with Stable Feature {self._stable_features[0]}:"
            else:
                sf_str = ""
                for sf in self._stable_features:
                    if sf_str:
                        sf_str = f"{sf_str}, {sf}"
                    else:
                        sf_str = sf
                metric_name = f"{metric_name} with Stable Features {sf_str}:"

        # a map of feature combinations to density_distributions
        feature_combinations = self.get_feature_combinations()
        feature_map_a = self.compute_combinations_density_distributions(feature_combinations, data_frame_a, weight_a)
        feature_map_b = self.compute_combinations_density_distributions(feature_combinations, data_frame_b, weight_b)
        baseline_feature_map = self.compute_combinations_density_distributions(feature_combinations,
                                                                               baseline_data_frame,
                                                                               baseline_weight) \
            if baseline_data_frame is not None else None

        # Get mapping results for the distributions
        density_dist_a_result = result.MappingResult(
            value={k: result.PandasResult(value=v, metric_name=metric_name) for k, v in feature_map_a.items()},
            metric_name=metric_name, level=ResultLevel.DETAIL,
            description='feature combinations to density distribution for data set a')
        density_dist_b_result = result.MappingResult(
            value={k: result.PandasResult(value=v, metric_name=metric_name) for k, v in feature_map_b.items()},
            metric_name=metric_name, level=ResultLevel.DETAIL,
            description='feature combinations to density distribution for data set b')
        density_dist_baseline_result = result.MappingResult(
            value={k: result.PandasResult(value=v, metric_name=metric_name)
                   for k, v in baseline_feature_map.items()},
            metric_name=metric_name, level=ResultLevel.DETAIL,
            description='feature combinations to density distribution for baseline data set') \
            if baseline_data_frame is not None else None

        density_dist_diffs = compute_mapping_feature_combinations_to_density_distribution_differences(
            feature_map_a, feature_map_b)
        density_dist_diffs_baseline = None
        if baseline_data_frame is not None:
            density_dist_diffs_baseline = compute_mapping_feature_combinations_to_density_distribution_differences(
                feature_map_a, baseline_feature_map)
            for k, df in density_dist_diffs.items():
                density_dist_diffs_baseline[k] = density_dist_diffs_baseline[k].reindex(index=df.index, fill_value=0)

        sf_result = None
        if self._stable_features:
            sf = StableFeatures(names=self._stable_features)
            if baseline_data_frame is not None:
                for k, df in density_dist_diffs.items():
                    sf.add_scores(df, density_dist_diffs_baseline[k])
            else:
                for df in density_dist_diffs.values():
                    sf.add_scores(df)
            sf_result = StableFeatureResult(value=sf.scores(), metric_name=metric_name, level=ResultLevel.SUMMARY,
                                            description='StableFeature_' + ('_'.join(self._stable_features)),
                                            sf=[self.get_feature_name(sf) for sf in self._stable_features],
                                            data_a_count=self.data_frame_a.shape[0],
                                            data_b_count=self.data_frame_b.shape[0],
                                            baseline_count=baseline_data_frame.shape[0]
                                            if baseline_data_frame is not None else 0)

        calculated_density_dist_diffs = compute_density_distribution_differences(density_dist_diffs)
        mean_density_dist_diffs = compute_density_mean(calculated_density_dist_diffs)

        individual_scores_result = result.ListResult(
            value=calculated_density_dist_diffs,
            metric_name=metric_name, level=ResultLevel.GENERAL,
            description=f'Individual scores for {str(self._marginal_dimensionality)}-marginal')

        per_value_tuple_result = result.ParquetResult(
            value=density_dist_diffs, level=ResultLevel.DETAIL,
            metric_name=metric_name,
            description=f'Per-value-tuple (sub-individual) scores for {str(self._marginal_dimensionality)}-marginal'
        )

        k_marginal_score = {
            'Raw Marginal Score': mean_density_dist_diffs,
            'Calculated Marginal Score': self._scale_raw_marginal_score(mean_density_dist_diffs)
        }
        duration = datetime.datetime.now() - start_time

        header_text: str = 'Summary:\n' \
                           'This report was generated using the k Marginal Metric approach\n' \
                           '- picking strategy: {picking}\n' \
                           '- number of marginals: {sample}\n\n' \
                           'The evaluation was run on {starttime} and took {duration} seconds.\n' \
                           'How to interpret raw results - the marginal metric score (value between 0 and 2):\n' \
                           '   0 - perfectly matching density distributions ' \
                           '(for the marginals used in the comparison).\n' \
                           '   2 - no overlap whatsoever (for the marginals used in the comparison).\n\n' \
                           'How to interpret Calculated Marginal Score results \n' \
                           '- the marginal metric score (value between 0 and 1000):\n' \
                           '   0 - no overlap whatsoever\n' \
                           '   1000 - perfectly matching density distributions\n'

        results = []
        score_result = result.MappingResult(value=k_marginal_score, metric_name=metric_name,
                                            description=header_text.format(
                                                picking=self._picking_strategy.name, sample=len(feature_map_a),
                                                starttime=start_time, duration=duration.total_seconds()),
                                            level=ResultLevel.SUMMARY)
        score_result.display_number_lines = 0
        results.append(score_result)

        combination_diff_result = None
        if baseline_data_frame is not None:
            baseline_diffs = compute_density_distribution_differences(density_dist_diffs_baseline)
            baseline_mean_density_dist_diffs = compute_density_mean(baseline_diffs)
            baseline_k_marginal_score = {
                'Baseline Raw Marginal Score': baseline_mean_density_dist_diffs,
                'Baseline Calculated Marginal Score': self._scale_raw_marginal_score(baseline_mean_density_dist_diffs)
            }
            baseline_score_result = result.MappingResult(value=baseline_k_marginal_score, metric_name=metric_name,
                                                         description=f"Baseline {self._baseline:.2f} Score",
                                                         level=ResultLevel.SUMMARY)
            baseline_score_result.display_number_lines = 0
            results.append(baseline_score_result)

            diff_index = [str(diff[0]) for diff in calculated_density_dist_diffs]
            diffs_df = pd.DataFrame(data=diff_index, columns=['combination'])
            diffs_df['score'] = [diff[1] for diff in calculated_density_dist_diffs]
            diffs_df['bl score'] = [diff[1] for diff in baseline_diffs]
            diffs_df['difference'] = diffs_df['bl score'] - diffs_df['score']
            combination_diff_result = TableResult(value=diffs_df, sort_column='difference', ascending=True,
                                                  description=f"Combination difference scores",
                                                  metric_name=metric_name, level=ResultLevel.SUMMARY)

        results.append(sf_result)
        frequent_item_set_df = self._compute_item_sets(calculated_density_dist_diffs)
        if not frequent_item_set_df.empty:
            frequent_item_result = TableResult(value=frequent_item_set_df, sort_column='count', ascending=False,
                                               metric_name=metric_name, description='frequent itemset',
                                               level=ResultLevel.GENERAL)
            results.append(frequent_item_result)
        results.append(density_dist_a_result)
        results.append(density_dist_b_result)
        if density_dist_baseline_result is not None:
            results.append(density_dist_baseline_result)
        results.append(individual_scores_result)

        if combination_diff_result is not None:
            results.append(combination_diff_result)
        results.append(per_value_tuple_result)

        to_return_result = result.ListResult(
            value=results,
            metric_name=metric_name,
            description='All of the results for the marginal metric computation')

        logging.info('compute_results complete')
        return to_return_result

    def _compute_item_sets(self, individual_marginal_scores: List[Tuple[tuple, float]]) -> pd.DataFrame:
        """
        This function computes the frequent item sets in the top percentage of worst calculated 3 marginals.

        :param: individual_marginal_scores: The list of individual combinations with its score.
        :return: the results of the frequent item set analysis.
        """
        logging.debug(msg=f"MarginalMetric._compute_item_sets().")

        individual_marginal_scores.sort(key=lambda x: x[1], reverse=True)
        count = math.ceil(len(individual_marginal_scores) * self.item_set_percentage)
        percentage_of_items = individual_marginal_scores[:count]
        item_set_df = frequent_item_set(percentage_of_items, self.min_support_item_set)
        if self._stable_features:
            mask = pd.Series(data=True, index=item_set_df.index)
            for index, row in item_set_df.iterrows():
                if all(item in row['itemsets'] for item in self._stable_features):
                    mask.iat[index] = False
            item_set_df = item_set_df[mask]
        return item_set_df
