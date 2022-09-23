import json
import logging
import os.path
from functools import partial
from typing import Dict, List

import pandas as pd

from censyn.features import Feature
from censyn.metrics import Metrics
from censyn.programs.censyn_base import CenSynBase
from censyn.report import ReportLevel
from censyn.results import ResultLevel, MappingResult
from ..utils.config_util import config_boolean, config_list, config_filters
from censyn.utils import pool_helper, find_class, calculate_weights

# Configuration labels
PROCESS_FEATURES: str = 'process_features'
IGNORE_FEATURES: str = 'ignore_features'
WEIGHT_FEATURES: str = 'weight_features'
DATA_FILES_A: str = 'data_files_a'
DATA_FILES_B: str = 'data_files_b'
FILTERS_A: str = 'filters_a'
FILTERS_B: str = 'filters_b'
BIN_USE_DATA_A: str = 'bin_use_data_a'
METRICS: str = 'metrics'
ALL_CONFIG_ITEMS = [PROCESS_FEATURES, IGNORE_FEATURES, WEIGHT_FEATURES,
                    DATA_FILES_A, DATA_FILES_B, FILTERS_A, FILTERS_B,
                    BIN_USE_DATA_A, METRICS]


def process_bin_series(feature: Feature, in_series: pd.Series) -> pd.DataFrame:
    """
    Helper function to call the feature transformations

    :param: feature: The feature object to be transformed
    :param: in_series: The Pandas Series of that feature
    :return: A transformed DataFrame
    """
    return feature.transform(in_series.to_frame(name=feature.feature_name))


class Evaluate(CenSynBase):
    def __init__(self, config_file: str) -> None:
        """
        Class to run an evaluation of a synthesized data set by creating metrics that compare the original and
        synthesized data sets.
        Can instantiate an arbitrary number of arbitrary Metrics, specified in the config file.

        :param: Configuration for the evaluation
        """
        super().__init__()
        logging.debug(msg=f"Instantiate Evaluate with config_file {config_file}.")

        self._data_files_a = None
        self._data_files_b = None
        self._process_features = None
        self._ignore_features = []
        self._additional_features = []
        self._weight_features = []
        self._metrics_config = []

        self._filters_a = []
        self._filters_b = []

        # These need to be separate since binning is optional on the Metric level now
        self._df_a = None
        self._df_b = None
        self._binned_df_a = None
        self._binned_df_b = None
        self._add_df_a = None
        self._add_df_b = None
        self._bin_use_data_a = True

        # Evaluation Metrics
        self._metrics: List[Metrics] = []

        # Load the configuration data
        if config_file:
            with open(config_file, 'r') as w:
                config = json.loads(w.read())
                self.load_config(config)

        # Create report
        self.initialize_report()

    @property
    def data_files_a(self) -> List[str]:
        return self._data_files_a

    @property
    def data_files_b(self) -> List[str]:
        return self._data_files_b

    @property
    def metrics(self) -> List[Metrics]:
        return self._metrics

    @property
    def all_config_items(self) -> List[str]:
        """Getter for the list of configuration items."""
        config_items = ALL_CONFIG_ITEMS.copy()
        config_items.append(super().all_config_items)
        return config_items

    def create_features(self, feature_def: List[Feature], features_name: str = "Features") -> Dict:
        """
        Create the Features' dictionary from a list for feature definitions.

        :param: feature_def: List of feature definitions.
        :param: features_name: The name of the features.
        :return: Dictionary of Features.
        """
        logging.debug(msg=f"Evaluate.create_features(). feature_def {feature_def} and features_name {features_name}.")
        # Create the features and verify they are in process_features
        if self._process_features:
            all_features = set(self._process_features + self._ignore_features + self._weight_features)
            all_def = [f for f in feature_def if f.feature_name in all_features]
            features = super().create_features(feature_def=all_def)

            for feature_name in self._ignore_features:
                if feature_name in self._process_features:
                    logging.warning(f"Ignore feature {feature_name} is already defined.")
            for feature_name in self._weight_features:
                if feature_name in self._process_features:
                    logging.warning(f"Weight feature {feature_name} is already defined.")
        else:
            features = super().create_features(feature_def=feature_def)
            self._process_features = [f for f in features.keys()
                                      if f not in self._weight_features and f not in self._ignore_features]

        for feature_name in self._weight_features:
            if feature_name in self._ignore_features:
                logging.warning(f"Weight feature {feature_name} is already defined in ignore features.")
        return features

    def _initialize_metrics(self) -> None:
        logging.debug(msg=f"Evaluate._initialize_metrics().")
        for metric_value in self._metrics_config:
            # Create each metric
            params = metric_value['attributes']
            params['features'] = self.features
            use_bins = metric_value.get('use_bins', None)
            if use_bins is not None:
                params['use_bins'] = use_bins
            use_weights = metric_value.get('use_weights', None)
            if use_weights is not None:
                params['use_weights'] = use_weights
            cur_metric = find_class(Metrics, metric_value['class'], params)
            self._metrics.append(cur_metric)

    def create_feature_bins(self) -> None:
        """
        Create the features' binner bins.

        :return: None
        """
        for f_name, feature in self.features.items():
            bin_df = self._df_a if self._bin_use_data_a else self._df_b
            feat_s = None if f_name not in bin_df else bin_df[f_name]
            if feature.binner is not None and feature.binner.bin_list is None:
                feature.binner.create_bins(in_s=feat_s)

    def _transform_features(self) -> None:
        """Prepare to apply binning to the data features"""
        logging.debug(msg=f"Evaluate._transform_features().")
        for metric in self._metrics:
            if metric.use_bins:
                logging.info(f'Transforming {len(self._features)} features for data set a')
                self._binned_df_a = self._transform_bin_data(self._df_a)
                logging.info(f'Transforming {len(self._features)} features for data set b')
                self._binned_df_b = self._transform_bin_data(self._df_b)
                break

    def _transform_bin_data(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin all the features' data.

        :param: in_df: The Pandas DataFrame of the  data.
        :return: Binned DataFrame.
        """
        logging.debug(msg=f"Evaluate._transform_bin_data(). in_df {in_df.shape}.")
        function_list = []
        for f_name, feature in self.features.items():
            if f_name not in in_df:
                if f_name not in self._weight_features:
                    logging.warning(f"Feature {f_name} is missing from at least one of the data frames.")
            else:
                function_list.append(partial(process_bin_series,
                                             feature=feature, in_series=in_df[f_name]))

        parallel = True if self.processes > 1 else False
        if parallel:
            # Process pool
            results = pool_helper(self.processes, function_list)
        else:
            results = (task() for task in function_list)
        return pd.concat(results, axis=1, sort=False)

    def execute(self) -> None:
        """Run the evaluation"""
        logging.debug(msg=f"Evaluate.execute().")
        self.load_features()
        self.validate_features()

        self._initialize_metrics()
        self.validate_metrics()

        self._df_a = self.load_data(self.data_files_a, features=self.features, data_name="Data A")
        self._df_b = self.load_data(self.data_files_b, features=self.features, data_name="Data B")

        self._df_a = self.generate_data(data_df=self._df_a, features=self.features, data_name="Data A")
        self._df_b = self.generate_data(data_df=self._df_b, features=self.features, data_name="Data B")

        self._df_a = self.filter_data(filters=self._filters_a, data_df=self._df_a, data_name="Data A")
        self._df_b = self.filter_data(filters=self._filters_b, data_df=self._df_b, data_name="Data B")

        # Weight
        weight_data_a = calculate_weights(in_df=self._df_a, features=self._weight_features)
        weight_data_b = calculate_weights(in_df=self._df_b, features=self._weight_features)

        self.create_feature_bins()
        self.report.time_function('transform_features-time', self._transform_features, {})

        logging.info('Starting metric evaluation')
        for metric in self.metrics:
            data_a = self._binned_df_a if metric.use_bins else self._df_a
            data_a = data_a[self._process_features]
            data_b = self._binned_df_b if metric.use_bins else self._df_b
            data_b = data_b[self._process_features]
            weight_a = weight_data_a if metric.use_weights else None
            weight_b = weight_data_b if metric.use_weights else None
            add_features = metric.additional_features
            add_a = self._df_a[add_features] if add_features else None
            add_b = self._df_b[add_features] if add_features else None
            cur_result = metric.compute_results(data_frame_a=data_a, data_frame_b=data_b,
                                                weight_a=weight_a, weight_b=weight_b,
                                                add_a=add_a, add_b=add_b)
            # Create the metric output report
            logging.info(f"Create {metric.name} report")
            append_fn = f"_{metric.name}"
            report_file_path = self.report.full_file_path
            if report_file_path:
                file_path, file_name = os.path.split(report_file_path)
                root_name, ext = os.path.splitext(file_name)
                report_file_path = os.path.join(file_path, root_name)
            cur_report = self.create_report(file_path=report_file_path, file_append=append_fn)
            cur_report.add_result(metric.name, cur_result)
            cur_report.features = self.features
            cur_report.produce_report()
            self.report.add_result(metric.name, cur_result)

        # Configuration result
        config_result = MappingResult(value=self.get_config(), level=ResultLevel.GENERAL,
                                      metric_name="Evaluate configuration")
        config_result.display_number_lines = 0
        self.report.add_result("Evaluate configuration", config_result)

        logging.info('Creating report')
        self.report.features = self.features
        self.report.save_files = False
        self.report.level = ReportLevel.SUMMARY
        self.report.produce_report()
        logging.info('Finish evaluate')

    def validate_features(self) -> None:
        """Check for valid dependencies for the features."""
        logging.debug(msg=f"Evaluate.validate_features().")
        super().validate_features(self.features)
        self.validate_filter_features(filters=self._filters_a, features=self.features)
        self.validate_filter_features(filters=self._filters_b, features=self.features)

        self.validate_list_features(self._process_features, self.features, feature_name="Process Features")
        self.validate_list_features(self._ignore_features, self.features, feature_name="Ignore Features")
        self.validate_list_features(self._weight_features, self.features, feature_name="Weight Features")

    def validate_metrics(self) -> None:
        """Check for valid dependencies for the metrics."""
        logging.debug(msg=f"Evaluate.validate_metrics().")
        for metric in self.metrics:
            self.validate_list_features(metric.additional_features, self.features,
                                        feature_name=f"{metric.name} additional Features")

            in_process_f = metric.in_process_features
            for cur_f in in_process_f:
                if cur_f not in self._process_features:
                    msg = f"{metric.name} Feature {cur_f} is not in the process features."
                    logging.error(msg=msg)
                    raise ValueError(msg)

    def _get_metrics_config(self) -> List:
        """
        Get the configuration of the metrics.

        :return: List of the configuration for each metric
        """
        metrics_config = []
        for cur_metric in self._metrics_config:
            # Create each metric
            params = cur_metric['attributes']
            features = params.get('features', None)
            if features:
                del params['features']
            metrics_config.append(cur_metric)
        return metrics_config

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        :return: Configuration dictionary:
        """
        logging.debug(msg=f"Evaluate.get_config().")
        config = {
            PROCESS_FEATURES: self._process_features,
            IGNORE_FEATURES: self._ignore_features,
            WEIGHT_FEATURES: self._weight_features,
            DATA_FILES_A: self._data_files_a,
            DATA_FILES_B: self._data_files_b,
            FILTERS_A: [{'class': f.__class__.__name__, 'attributes': f.to_dict()} for f in self._filters_a],
            FILTERS_B: [{'class': f.__class__.__name__, 'attributes': f.to_dict()} for f in self._filters_b],
            BIN_USE_DATA_A: self._bin_use_data_a,
            METRICS: self._get_metrics_config(),
        }
        config.update(super().get_config())
        return config

    def save_config(self, config_file: str) -> None:
        """
        DEPRECATED
        Save a configuration file.

        :param config_file:
        """
        if config_file:
            with open(config_file, 'w') as outfile:
                json.dump(self.get_config(), outfile, indent=4)

    def load_config(self, config: Dict) -> None:
        """
        Loads the config...

        :param: config: the loaded config
        """
        logging.debug(msg=f"Evaluate.load_config(). config {config}.")
        super_config = {}
        for key, v in config.items():
            if key not in ALL_CONFIG_ITEMS:
                if key in super().all_config_items:
                    super_config[key] = v
                else:
                    msg = f"Non-supported key {key} in the configuration."
                    logging.debug(msg=msg)
                    raise ValueError(msg)
        super().load_config(config=super_config)

        self._process_features = config_list(config=config, key=PROCESS_FEATURES, data_type=str, default_value=[])
        self._ignore_features = config_list(config=config, key=IGNORE_FEATURES, data_type=str, default_value=[])
        self._weight_features = config_list(config=config, key=WEIGHT_FEATURES, data_type=str, default_value=[])
        self._data_files_a = config_list(config=config, key=DATA_FILES_A, data_type=str, default_value=[])
        self._data_files_b = config_list(config=config, key=DATA_FILES_B, data_type=str, default_value=[])

        # Configure the filters
        self._filters_a = config_filters(config=config, key=FILTERS_A, default_value=[])
        self._filters_b = config_filters(config=config, key=FILTERS_B, default_value=[])

        self._bin_use_data_a = config_boolean(config=config, key=BIN_USE_DATA_A, default_value=None)

        # Metrics configuration
        self._metrics_config = config_list(config=config, key=METRICS, data_type=Dict, default_value=None)
        if self._metrics_config is None:
            logging.error(msg="No Metrics specified")
            raise ValueError("No Metrics specified")
