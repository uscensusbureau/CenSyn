import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

import censyn.report as rp
from censyn.datasources import load_data_source, FeatureJsonFileSource
from censyn.features import set_data_type, Feature
from censyn.filters import Filter
from ..utils.config_util import config_int, config_string, config_dict, config_logging_level
from ..utils.file_util import append_to_file_name
from ..utils.logger_util import LOG_FORMAT, get_logging_level, set_logging_level, get_logging_file, set_logging_file

# Configuration labels
LOGGING_LEVEL: str = 'logging_level'
LOGGING_FILE: str = 'logging_file'
NUM_PROCESSES: str = 'processes'
RANDOM_SEED: str = 'random_seed'
FEATURE_FILE: str = 'features_file'
REPORT_CONFIG: str = 'report'
ALL_CONFIG_ITEMS = [LOGGING_LEVEL, LOGGING_FILE, NUM_PROCESSES, RANDOM_SEED, FEATURE_FILE, REPORT_CONFIG]


class CenSynBase(ABC):
    """Base CenSyn process class."""
    def __init__(self) -> None:
        """Initialize the CenSynBase."""
        logging.debug(msg=f"Instantiate CenSynBase .")
        self._processes: int = 1
        self._random_seed = None

        # Features
        self._features_file = None
        self._features = None

        # Report
        self._report_config = None
        self._report = None

    @property
    def processes(self) -> int:
        """Getter for processes property."""
        return self._processes

    @property
    def random_seed(self) -> int:
        """Getter for random seed property."""
        return self._random_seed

    @property
    def features(self) -> Dict:
        """features getter"""
        return self._features

    @property
    def report(self) -> rp.Report:
        """Getter for report property."""
        return self._report

    @property
    def all_config_items(self) -> List[str]:
        """Getter for the list of configuration items."""
        return ALL_CONFIG_ITEMS

    def initialize_report(self) -> None:
        """Create and set the report."""
        self._report = self.create_report()

    def create_report(self, file_path: str = None, file_append: str = "") -> rp.Report:
        """
        Create report for the process.

        :param: file_path: Path to write the output report.
        :param: file_append: String to append to file name.
        :return: FileReport or ConsoleReport
        """
        logging.debug(msg=f"CenSynBase.create_report(). file_path {file_path} and file_append {file_append}.")
        r_file = self._report_config.get('report_file', None)
        if r_file:
            if file_path:
                r_path, r_name = os.path.split(r_file)
                r_file = os.path.join(file_path, r_name)
            file_name = append_to_file_name(file_name=r_file, file_append=file_append)
            report = rp.FileReport(config=self._report_config, file_full_path=file_name, rename_if_exists=True)
        else:
            report = rp.ConsoleReport(config=self._report_config)
        return report

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        :return: configuration dictionary:
        """
        config = {
            LOGGING_LEVEL: get_logging_level()
        }
        if get_logging_file():
            config[LOGGING_FILE] = get_logging_file()

        config.update({
            NUM_PROCESSES: self._processes,
            RANDOM_SEED: self._random_seed,
            FEATURE_FILE: self._features_file,
            # Report
            REPORT_CONFIG: self._report_config
        })
        return config

    def load_config(self, config: Dict):
        """
        Load the configuration.

        :param: config: Dictionary of configurations.
        :return: None
        """
        logging.debug(msg=f"CenSynBase.load_config(). config {config}.")

        # Verify the keys
        for key in config.keys():
            if key not in ALL_CONFIG_ITEMS:
                msg = f"Non-supported key {key} in the configuration."
                logging.debug(msg=msg)
                raise ValueError(msg)

        # Set logger
        set_logging_level(config_logging_level(config=config, key=LOGGING_LEVEL,
                                               default_value=get_logging_level()))
        file_name = config_string(config=config, key=LOGGING_FILE, default_value=None)
        if file_name:
            set_logging_file(file_name=file_name)
            root = logging.getLogger()
            fh = logging.FileHandler(filename=get_logging_file())
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(fh)

        # Set number of processes
        self._processes = config_int(config=config, key=NUM_PROCESSES, default_value=self._processes)

        # Set Random Seed
        self._random_seed = config_int(config=config, key=RANDOM_SEED, default_value=self._random_seed)
        if self._random_seed:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)

        self._features_file = config_string(config=config, key=FEATURE_FILE, default_value=None)

        # Report
        self._report_config = config_dict(config=config, key=REPORT_CONFIG, default_value=self._report_config)
        if self._report_config is None:
            msg = f"No Report configuration specified."
            logging.debug(msg=msg)
            raise ValueError(msg)

    @staticmethod
    def validate_features(features: Dict) -> None:
        """
        Validate the features.

        :param: features: The dictionary of features.
        :raises: ValueError on missing feature.
        """
        logging.debug(msg=f"CenSynBase.validate_features(). features {features}.")
        for feat in features.values():
            if len(feat.dependencies) > 0:
                for dep_f in feat.dependencies:
                    if dep_f not in features:
                        msg = f"Feature {feat} has dependency of {dep_f} which does not exist."
                        logging.error(msg=msg)
                        raise ValueError(msg)
            if len(feat.exclude_dependencies) > 0:
                for dep_f in feat.exclude_dependencies:
                    if dep_f not in features:
                        msg = f"Feature {feat} has exclude dependency of {dep_f} which does not exist."
                        logging.error(msg=msg)
                        raise ValueError(msg)

    @staticmethod
    def validate_filter_features(filters: List[Filter], features: Dict) -> None:
        """
        validate the feature dependencies for the filters.

        :param: filters: List of filters.
        :param: features: The dictionary of features.
        :raises: ValueError on missing feature.
        """
        logging.debug(msg=f"CenSynBase.validate_filter_features(). filters {filters} and features {features}.")
        for cur_f in filters:
            dep = cur_f.dependency
            if len(dep) > 0:
                for dep_f in dep:
                    if dep_f not in features:
                        msg = f"Filter {cur_f} has dependency of {dep_f} which does not exist."
                        logging.error(msg=msg)
                        raise ValueError(msg)

    @staticmethod
    def validate_list_features(feat_l: List[str], features: Dict, feature_name: str = "Feature") -> None:
        """
        validate the feature dependencies for the filters.

        :param: feat_l: List of feature names.
        :param: features: The dictionary of features.
        :param: feature_name: String name for the feature.
        :raises: ValueError on missing feature.
        """
        logging.debug(msg=f"CenSynBase.validate_list_features(). feat_l {feat_l}, features {features} and "
                          f"feature_name {feature_name}.")
        for cur_f in feat_l:
            if cur_f not in features:
                msg = f"{feature_name} {cur_f} is not in the features."
                logging.error(msg=msg)
                raise ValueError(msg)

    @staticmethod
    def load_data(file_names: List, features: Dict = None, data_name: str = "Data") -> pd.DataFrame:
        """
        Loads the input data for processing.

        :param: file_names: The list of file names for the data source.
        :param: features: The dictionary of features
        :param: data_name: String name for the dataset.
        :return: The DataFrame for processing
        """
        logging.debug(msg=f"CenSynBase.load_data(). file_names {file_names}, features {features} and "
                          f"data_name {data_name}.")
        data_df = pd.DataFrame()
        if file_names:
            if features:
                data_df = set_data_type(in_df=load_data_source(file_names, list(features.keys())), features=features)
            else:
                data_df = load_data_source(file_names)
        logging.info(f'Load {data_name} size {data_df.shape[0]} rows by {data_df.shape[1]} columns')
        return data_df

    @staticmethod
    def generate_data(data_df: pd.DataFrame, features: Dict, data_name: str = "Data") -> pd.DataFrame:
        """
        Generate data from feature functions.

        :param: data_df: The DataFrame to base generation upon.
        :param: features: The dictionary of features
        :param: data_name: String name for the dataset.
        :return: The generated DataFrame.
        """
        logging.debug(msg=f"CenSynBase.generate_data(). data_df {data_df}, features {features} and "
                          f"data_name {data_name}.")
        modify = False
        if features:
            for feat in features.values():
                if feat.feature_name not in data_df.columns:
                    calc_s = feat.calculate_feature_data(data_df=data_df)
                    if isinstance(calc_s, pd.Series):
                        modify = True
                        data_df[feat.feature_name] = calc_s

        # Set order of columns the order of the features.
        if modify:
            columns = [c.feature_name for c in features.values() if c.feature_name in data_df.columns]
            data_df = data_df[columns]
            logging.info(f'Generate {data_name} size {data_df.shape[0]} rows by {data_df.shape[1]} columns')

        return data_df

    @staticmethod
    def filter_data(filters: List[Filter], data_df: pd.DataFrame, data_name: str = "Data") -> pd.DataFrame:
        """
        filters the data.

        :param: filters: List of filters to perform on the data.
        :param: data_df: The DataFrame to filter.
        :param: data_name: String name for the dataset.
        :return: The filtered DataFrame.
        """
        logging.debug(msg=f"CenSynBase.filter_data(). filters {filters}, data_df {data_df} and "
                          f"data_name {data_name}.")
        # Filter the data
        for f in filters:
            logging.info(f'Executing {f.__class__.__name__} on data.')
            data_df = f.execute(data_df)
        logging.info(f'Filter {data_name} size {data_df.shape[0]} rows by {data_df.shape[1]} columns')
        return data_df

    @abstractmethod
    def execute(self) -> None:
        """Execute the CenSyn process."""
        raise NotImplementedError('Attempted call to abstract method CenSynBase.execute()')

    def load_features(self) -> None:
        """
        Load the features from the feature definition file.

        :return: None
        """
        logging.debug(msg=f"CenSynBase.load_features(). ")
        feature_def = self.create_feature_definitions(self._features_file)
        self._features = self.create_features(feature_def)

    @staticmethod
    def create_feature_definitions(features_file: str, features_name: str = "Features") -> List[Feature]:
        """
        Load a feature definition from a features file.

        :param: features_file: The file of the features definitions.
        :param: features_name: The name of the features.
        :return: List of features definitions.
        """
        logging.debug(msg=f"CenSynBase.create_feature_definitions(). features_file {features_file} and "
                          f"features_name {features_name}.")
        if features_file:
            logging.info(f'Loading {features_name} file {features_file}')
            try:
                return FeatureJsonFileSource(features_file).feature_definitions
            except ValueError as e:
                logging.error(f"Unable to load {features_file}")
                raise ValueError(e)
        return []

    def create_features(self, feature_def: List[Feature], features_name: str = "Features") -> Dict:
        """
        Create the Features' dictionary from a list for feature definitions.

        :param: feature_def: List of feature definitions.
        :param: features_name: The name of the features.
        :return: Dictionary of Features.
        """
        logging.debug(msg=f"CenSynBase.create_features(). feature_def {feature_def} and "
                          f"features_name {features_name}.")
        features = {feature.feature_name: feature for feature in feature_def}
        logging.info(f"Create {features_name} size is {len(features)}")
        return features
