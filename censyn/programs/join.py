import datetime
import json
import logging
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pandas as pd

from censyn.checks import rename_variables
from censyn.datasources import save_data_source, validate_data_source, FeatureJsonFileSource
from censyn.features import Feature
from censyn.filters import Filter
from censyn.results import ResultLevel, MappingResult, StrResult
from .censyn_base import CenSynBase
from ..utils.config_util import config_boolean, config_string, config_list, config_filters

HEADER_TEXT: str = 'Join Summary: \n' \
                   'The join was run on {start_time} and took {duration} seconds. \n' \
                   'Output data size is {columns} columns by {rows} rows. \n'


# Configuration labels
INPUT_DATA_FILES: str = 'input_data_files'
INPUT_FILTERS: str = 'input_filters'
INPUT_SUBSETS: str = 'input_subsets'
INPUT_SUBSETS_MERGE_HOW: str = 'input_subset_merge_how'
JOIN_FEATURES_FILE: str = 'join_features_file'
JOIN_DATA_FILES: str = 'join_data_files'
JOIN_FILTERS: str = 'join_filters'
JOIN_SUBSETS: str = 'join_subsets'
JOIN_SUBSETS_MERGE_HOW: str = 'join_subset_merge_how'
MERGE_ON_FEATURES: str = 'merge_on_features'
JOIN_FEATURES_LIST: str = 'join_features_list'  # Deprecated
MERGE_HOW: str = 'merge_how'
ALLOW_DUPLICATES: str = 'allow_duplicates'
INPUT_APPEND_STR: str = 'input_append_str'
JOIN_APPEND_STR: str = 'join_append_str'
OUTPUT_FEATURE_FILE: str = 'output_features_file'
OUTPUT_FILE: str = 'output_file'
ALL_CONFIG_ITEMS = [INPUT_DATA_FILES, INPUT_FILTERS, INPUT_SUBSETS, INPUT_SUBSETS_MERGE_HOW,
                    JOIN_FEATURES_FILE, JOIN_DATA_FILES, JOIN_FILTERS, JOIN_SUBSETS, JOIN_SUBSETS_MERGE_HOW,
                    MERGE_ON_FEATURES, JOIN_FEATURES_LIST, MERGE_HOW, ALLOW_DUPLICATES,
                    INPUT_APPEND_STR, JOIN_APPEND_STR, OUTPUT_FEATURE_FILE, OUTPUT_FILE]


class JoinSubset:
    def __init__(self, filters: List[Filter], append_str: str, name: str) -> None:
        self._filters = filters
        self._append_str = append_str
        self._name = name

    @property
    def filters(self) -> List[Filter]:
        """Getter for filters"""
        return self._filters

    @property
    def append_str(self) -> str:
        """Getter for append_str"""
        return self._append_str

    @property
    def name(self) -> str:
        """Getter for name"""
        return self._name


class Join(CenSynBase):
    """Join two input data sets."""
    def __init__(self, config_file: str) -> None:
        """
        Initialization of Join.

        :param: config_file: Configuration file containing the parameters for join.
        """
        super().__init__()
        logging.debug(msg=f"Instantiate Join with config_file {config_file}.")

        self._input_data_files = []
        self._input_filters = []
        self._input_subsets: List[JoinSubset] = []
        self._input_subset_merge_how = "outer"

        self._join_features_file = None
        self._join_features = None
        self._join_data_files = []
        self._join_filters = []
        self._join_subsets: List[JoinSubset] = []
        self._join_subset_merge_how = "outer"

        self._merge_on_features = []
        self._merge_how = "inner"
        self._allow_duplicates = True
        self._input_append_str = "_X"
        self._join_append_str = "_Y"
        self._output_features_file = None
        self._output_file = ''

        if config_file:
            with open(config_file, 'r') as w:
                config = json.loads(w.read())
                self.load_config(config)
            logging.info(f'Load config file {config_file}')

        # Create report
        self.initialize_report()

        # Validate the settings
        self._valid_merge_how(merge_how=self._input_subset_merge_how, name='input_subset_merge_how')
        self._valid_merge_how(merge_how=self._join_subset_merge_how, name='join_subset_merge_how')
        self._valid_merge_how(merge_how=self._merge_how, name='merge_how')
        if self._input_subsets:
            self._valid_subsets(subsets=self._input_subsets, name="Input Subset")
        if self._join_subsets:
            self._valid_subsets(subsets=self._join_subsets, name="Join Subset")
        if not self._input_append_str and not self._join_append_str:
            msg = "Both input_append_str and join_append_str cannot be empty."
            logging.error(msg=msg)
            raise ValueError(msg)
        if self._input_append_str is not None and self._join_append_str is not None:
            if self._input_append_str == self._join_append_str:
                msg = f"input_append_str '{self._input_append_str}' cannot equal join_append_str " \
                      f"'{self._join_append_str}'."
                logging.error(msg=msg)
                raise ValueError(msg)

        if self._output_file:
            if not validate_data_source(self._output_file):
                msg = f"Must provide a supported output data file. Found {self._output_file} instead."
                logging.error(msg=msg)
                raise ValueError(msg)

    @property
    def all_config_items(self) -> List[str]:
        """Getter for the list of configuration items."""
        config_items = ALL_CONFIG_ITEMS.copy()
        config_items.append(super().all_config_items)
        return config_items

    def execute(self) -> None:
        """Execute the synthesize process."""
        logging.debug(msg=f"Join.execute().")
        e_start_time = datetime.datetime.now()

        # Load input features and data
        logging.info("Loading input files...")
        input_feature_json, self._features = self._load_feature_file(features_file=self._features_file,
                                                                     name="Input Features")
        if input_feature_json:
            self.validate_join_features(filters=self._input_filters, subsets=self._input_subsets,
                                        features=self.features)

            input_df = self.load_data(self._input_data_files, self.features, data_name="Input Data")
            logging.info("Generate the input data.")
            input_df = self.generate_data(data_df=input_df, features=self.features, data_name="Input Data")
            logging.info("Filter the input data.")
            input_df = self.filter_data(filters=self._input_filters, data_df=input_df, data_name="Input Data")
        else:
            logging.info("No Input features.")
            input_df = pd.DataFrame()

        # Load join features and data
        logging.info("Loading join files...")
        join_feature_json, self._join_features = self._load_feature_file(features_file=self._join_features_file,
                                                                         name="Join Features")
        if join_feature_json:
            self.validate_join_features(filters=self._join_filters, subsets=self._join_subsets,
                                        features=self._join_features)

            join_df = self.load_data(self._join_data_files, self._join_features, data_name="Join Data")
            logging.info("Generate the join data.")
            join_df = self.generate_data(data_df=join_df, features=self._join_features, data_name="Join Data")
            logging.info("Filter the join data.")
            join_df = self.filter_data(filters=self._join_filters, data_df=join_df, data_name="Join Data")
        else:
            logging.info("No Join features.")
            join_df = pd.DataFrame()

        self._validate(input_df=input_df, join_df=join_df)

        duplicate_features = list(set(input_df.columns).intersection(set(join_df.columns)))
        for feat in self._merge_on_features:
            if feat in duplicate_features:
                duplicate_features.remove(feat)
        if len(duplicate_features) == 0:
            logging.info("No duplicate features.")
        else:
            logging.info(f"Duplicate features {duplicate_features}.")

        if self._input_subsets and not input_df.empty:
            input_df, input_feature_json = self.report.time_function('merge-subsets-time', self._merge_subsets,
                                                                     {'subsets': self._input_subsets,
                                                                      'subset_merge_how': self._input_subset_merge_how,
                                                                      'merge_df': input_df,
                                                                      'feature_json': input_feature_json})

        if self._join_subsets and not join_df.empty:
            join_df, join_feature_json = self.report.time_function('merge-subsets-time', self._merge_subsets,
                                                                   {'subsets': self._join_subsets,
                                                                    'subset_merge_how': self._join_subset_merge_how,
                                                                    'merge_df': join_df,
                                                                    'feature_json': join_feature_json})

        output_df, output_features = self.report.time_function('merge-input-join-time',
                                                               self._merge_input_join,
                                                               {'input_df': input_df,
                                                                'input_feature_json': input_feature_json,
                                                                'join_df': join_df,
                                                                'join_feature_json': join_feature_json,
                                                                'duplicate_features': duplicate_features})

        # Save output files
        logging.info(f"Output features file {self._output_features_file}")
        with open(self._output_features_file, 'w') as f:
            json.dump(output_features, f, indent=4)

        input_path: Path = Path(self._features_file if self._features_file else self._join_features_file)
        output_path: Path = Path(self._output_features_file)
        schema_input_path = Path(input_path.parents[0]).joinpath(f'{input_path.stem}_schema{input_path.suffix}')
        schema_output_path = Path(output_path.parents[0]).joinpath(f'{output_path.stem}_schema{output_path.suffix}')
        shutil.copy(str(schema_input_path), str(schema_output_path))

        logging.info(f'Output Data size {output_df.shape[0]} rows by {output_df.shape[1]} columns')
        logging.info(f'Writing to output file {self._output_file}.')
        save_data_source(file_name=self._output_file, in_df=output_df)
        logging.info(f'Finished writing to output file {self._output_file}.')

        # Configuration result
        config_result = MappingResult(value=self.get_config(), level=ResultLevel.GENERAL,
                                      metric_name="Join configuration")
        self.report.add_result("Join configuration", config_result)

        # Create the output report
        logging.info('Create report')
        e_duration = datetime.datetime.now() - e_start_time
        self.report.header = HEADER_TEXT.format(start_time=e_start_time,
                                                duration=e_duration.total_seconds(),
                                                columns=output_df.shape[1],
                                                rows=output_df.shape[0])

        join_names = ", ".join([f'"{col}"' for col in output_df.columns])
        self.report.add_result(key="Names", value=StrResult(value=join_names, metric_name="join",
                                                            description="Names of the features"))
        self.report.produce_report()
        logging.info('Finish join')

    def validate_join_features(self, filters: List[Filter], subsets: List[JoinSubset], features: Dict) -> None:
        """
        validate the features and the dependencies for the filters.

        :param: filters: List of filters.
        :param: subsets: List of JoinSubset for merging.
        :param: features: The dictionary of features.
        :raises: ValueError on missing feature.
        """
        logging.debug(msg=f"Join.validate_join_features(). filters {filters}, subsets {subsets} and "
                          f"features {features}")
        """Check for valid dependencies for the features."""
        self.validate_features(features)
        self.validate_filter_features(filters=filters, features=features)
        for subset in subsets:
            self.validate_filter_features(filters=subset.filters, features=features)

        self.validate_list_features(self._merge_on_features, features, feature_name="Merge On Features")

    @staticmethod
    def _valid_merge_how(merge_how: str, name: str) -> None:
        """
        Validate the merge how.

        :param: merge_how: String value of how to merge,
        :param: name: Name of the merge.
        :return: None
        """
        logging.debug(msg=f"Join._valid_merge_how(). merge_how {merge_how} and name {name}")
        if merge_how not in ["left", "right", "outer", "inner", "cross"]:
            msg = f"Invalid {name} value of '{merge_how}'."
            logging.error(msg=msg)
            raise ValueError(msg)

    @staticmethod
    def _valid_subsets(subsets: List[JoinSubset], name: str) -> None:
        """
        Validate the subsets.

        :param: subsets: List of JoinSubset to validate.
        :param: name: Name of the subset.
        :return: None
        """
        logging.debug(msg=f"Join._valid_subsets(). subsets {subsets} and name {name}")
        for subset in subsets:
            if not subset.append_str:
                msg = f"Subset {name} append_str cannot be empty."
                logging.error(msg=msg)
                raise ValueError(msg)
        all_append_strings = list(subset.append_str for subset in subsets)
        for append_str, count in Counter(all_append_strings).items():
            if count > 1:
                msg = f"Subset append_str '{append_str}' occurs {count} times. It must be unique."
                logging.error(msg=msg)
                raise ValueError(msg)

    def _validate(self, input_df: pd.DataFrame, join_df: pd.DataFrame) -> None:
        """
        Validate the data.

        :param: input_df: Dataframe of the input data.
        :param: join_df: Dataframe of the join data.
        :return: None
        """
        logging.debug(msg=f"Join._validate(). input_df {input_df.shape} and join_df {join_df.shape}")
        if not self._merge_on_features:
            msg = f"Cannot have an empty merge_on_features."
            logging.error(msg=msg)
            raise ValueError(msg)
        if input_df.empty and join_df.empty:
            msg = f"Both the input and join data is empty()"
            logging.error(msg=msg)
            raise ValueError(msg)
        if not input_df.empty:
            if not set(self._merge_on_features).issubset(set(input_df.columns)):
                msg = f"The merge_on_features {self._merge_on_features} are not in the input dataset."
                logging.error(msg=msg)
                raise ValueError(msg)
        if not join_df.empty:
            if not set(self._merge_on_features).issubset(set(join_df.columns)):
                msg = f"The merge_on_features {self._merge_on_features} are not in the join dataset."
                logging.error(msg=msg)
                raise ValueError(msg)

    def _merge_subsets(self, subsets: List[JoinSubset], subset_merge_how: str, merge_df: pd.DataFrame,
                       feature_json: Dict[str, any]) -> (pd.DataFrame, Dict):
        """
        Merge the join subsets together.

        :param: subsets: List of JoinSubset for merging.
        :param: subset_merge_how: String of how th merge the subset data.
        :param: merge_df: Dataframe of the subset data to merge.
        :param: feature_json: The json feature definition for the data.
        :return: The merged data and json feature definition.
        """
        def _subset(subset: JoinSubset) -> (pd.DataFrame,  Dict[str, any]):
            """
            Create the Dataframe and json feature definition for the subset.

            :param: subset: The JoinSubset.
            :return: Dataframe and json feature definition for the subset.
            """
            subset_json = self._update_features_json(deepcopy(feature_json), subset.append_str, duplicate_features)
            subset_df = self.filter_data(filters=subset.filters, data_df=merge_df,
                                         data_name=f"Subset {subset.name} Data")
            return subset_df, subset_json

        logging.debug(msg=f"Join._merge_subsets(). subsets {subsets}, subset_merge_how {subset_merge_how}, "
                          f"merge_df {merge_df.shape} and feature_json {feature_json}")
        # Initialize the duplicate features
        duplicate_features = [col for col in merge_df.columns if col not in self._merge_on_features]

        # Prepare first subset
        output_df, output_json = _subset(subset=subsets[0])
        if len(subsets) <= 1:
            return output_df, feature_json

        # Merge the subsets together.
        for cur_subset in subsets[1:-1]:
            cur_df, cur_json = _subset(subset=cur_subset)
            output_json = {**output_json, **cur_json}
            output_df = output_df.merge(right=cur_df, how=subset_merge_how, on=self._merge_on_features,
                                        suffixes=(None, cur_subset.append_str))
            logging.info(f"Merge Subset {cur_subset.name} Data size {output_df.shape[0]} rows by "
                         f"{output_df.shape[1]} columns")
        last_df, last_json = _subset(subset=subsets[-1])
        output_json = {**output_json, **last_json}
        output_df = output_df.merge(right=last_df, how=subset_merge_how, on=self._merge_on_features,
                                    suffixes=(subsets[0].append_str, subsets[-1].append_str))

        logging.info(f'Merge Subset Data size {output_df.shape[0]} rows by {output_df.shape[1]} columns')
        return output_df, output_json

    def _merge_input_join(self, input_df: pd.DataFrame, input_feature_json: Dict,
                          join_df: pd.DataFrame, join_feature_json: Dict,
                          duplicate_features: List[str]) -> (pd.DataFrame, Dict):
        """
        Merge the input and join data.

        :param: input_df: Dataframe of the input data.
        :param: input_feature_json: The json feature definition for the input data.
        :param: join_df: Dataframe of the join data.
        :param: join_feature_json: The json feature definition for the join data.
        :param: duplicate_features: List of duplicate features.
        :return: The merged data and json feature definition.
        """
        logging.debug(msg=f"Join._merge_input_join(). input_df {input_df.shape}, "
                          f"input_feature_json {input_feature_json}, join_df {join_df.shape}, "
                          f"join_feature_json {join_feature_json} and duplicate_features {duplicate_features}")
        if input_df.empty:
            return join_df, join_feature_json
        if join_df.empty:
            return input_df, input_feature_json

        # Process duplicates when not allowed.
        if not self._allow_duplicates:
            for d_feat in duplicate_features:
                join_df.drop(d_feat, axis=1, inplace=True)

        # Perform merge of the two data frames
        input_suffix = f"{self._input_append_str}" if self._input_append_str else None
        join_suffix = f"{self._join_append_str}" if self._join_append_str else None
        output_df = input_df.merge(right=join_df, how=self._merge_how, on=self._merge_on_features,
                                   suffixes=(input_suffix, join_suffix))
        # Combine the two features definition dictionaries
        output_features = self.combine_features(input_feature_json=input_feature_json,
                                                join_feature_json=join_feature_json,
                                                duplicate_features=duplicate_features)
        return output_df, output_features

    def _load_feature_file(self, features_file: str, name: str = "Features") -> (Dict[str, any], Dict[str, Feature]):
        """
        Load the features from the features  file.

        :param: features_file: Name of features file.
        :param: name: Name
        :return: Feature json and Features dictionaries.
        """
        logging.debug(msg=f"Join._load_feature_file(). features_file {features_file} and name {name}")
        logging.info(f"Loading features file {features_file}")
        if features_file:
            json_fs = FeatureJsonFileSource(features_file)
            feature_json = json_fs.feature_json
            features = self.create_features(feature_def=json_fs.feature_definitions, features_name=name)
        else:
            feature_json = {}
            features = {}
        return feature_json, features

    def combine_features(self, input_feature_json:  Dict[str, any], join_feature_json:  Dict[str, any],
                         duplicate_features: List[str]) -> Dict:
        """
        Combine the features.

        :param: input_feature_json: The json feature definition for the input data.
        :param: join_feature_json: The json feature definition for the join data.
        :param: duplicate_features: List of features in both input and join datasets.
        :return: The json feature definition.
        """
        logging.debug(msg=f"Join.combine_features(). input_feature_json {input_feature_json}, "
                          f"join_feature_json {join_feature_json} and duplicate_features {duplicate_features}")
        # Process duplicate or common features between two set of features
        if not self._allow_duplicates:
            # Remove from the join feature dictionary
            for d_feat in duplicate_features:
                del join_feature_json[d_feat]
        else:
            input_feature_json = self._update_features_json(input_feature_json, self._input_append_str,
                                                            duplicate_features)
            join_feature_json = self._update_features_json(join_feature_json, self._join_append_str,
                                                           duplicate_features)

        # Save the combined features file
        output_features = {**input_feature_json, **join_feature_json}
        return output_features

    def _update_features_json(self, feature_json:  Dict[str, any], append_str: str,
                              duplicate_features: List[str]) -> Dict:
        """
        Update the features json definition with the append-string.

        :param: feature_json: The json feature definition for the data.
        :param: append_str: Append string.
        :param: duplicate_features: List of features in the dataset.
        :return: The json feature definition.
        """
        logging.debug(msg=f"Join._update_features_json(). feature_json {feature_json}, append_str {append_str} and "
                          f"duplicate_features {duplicate_features}")
        if append_str:
            rename_map = {dup_f: dup_f + append_str for dup_f in duplicate_features}
            self._update_features(feature_json, rename_map)
            for d_feat in duplicate_features:
                if d_feat in feature_json:
                    feature_json[d_feat + append_str] = feature_json[d_feat]
                    del feature_json[d_feat]
        return feature_json

    @staticmethod
    def _update_features(features: Dict[str, any], duplicates: Dict[str, str]) -> None:
        """
        Update the features.

        :param: features:  The json feature definition configuration.
        :param: duplicates: Dictionary of the mapping of duplicate names.
        :return: None
        """
        logging.debug(msg=f"Join._update_features(). features {features} and duplicates {duplicates}")
        # Validate the duplicates
        rename_variables(expression="", rename=duplicates, validate=True)

        # Process every feature
        for k, v in features.items():
            f_format = v.get('feature_format', None)
            if f_format and k in duplicates.keys():
                v['feature_format'] = rename_variables(f_format, duplicates)
            f_encoder = v.get('encoder', None)
            if f_encoder:
                encoder_param = f_encoder.get('encoder_param', None)
                if encoder_param:
                    column = encoder_param.get('column', None)
                    if column and column in duplicates.keys():
                        encoder_param['column'] = duplicates[column]
            f_model = v.get('model_info', None)
            if f_model:
                model_name = f_model.get('model_name', "")
                if model_name == "CalculateModel":
                    model_params = f_model.get('model_params', None)
                    if model_params:
                        model_params['expr'] = rename_variables(model_params.get('expr', ""), duplicates)
            f_dependencies = v.get('dependencies', None)
            if f_dependencies:
                new_dependencies = []
                for dep in f_dependencies:
                    if dep in duplicates.keys():
                        new_dependencies.append(duplicates[dep])
                v['dependencies'] = new_dependencies
            f_exclude_dependencies = v.get('exclude_dependencies', None)
            if f_dependencies:
                new_exclude_dependencies = []
                for dep in f_exclude_dependencies:
                    if dep in duplicates.keys():
                        new_exclude_dependencies.append(duplicates[dep])
                v['exclude_dependencies'] = new_exclude_dependencies

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        :return: configuration dictionary:
        """
        logging.debug(msg=f"Join.get_config().")
        config = {
            INPUT_DATA_FILES: self._input_data_files,
            INPUT_FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                            for f in self._input_filters],
            INPUT_SUBSETS: [{'subset_filters': [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                                                for f in subset.filters],
                             'subset_append_str': subset.append_str,
                             'subset_name': subset.name} for subset in self._input_subsets],
            INPUT_SUBSETS_MERGE_HOW: self._input_subset_merge_how,
            JOIN_FEATURES_FILE: self._join_features_file,
            JOIN_DATA_FILES: self._join_data_files,
            JOIN_FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()} for f in self._join_filters],
            JOIN_SUBSETS: [{'subset_filters': [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                                               for f in subset.filters],
                            'subset_append_str': subset.append_str,
                            'subset_name': subset.name} for subset in self._join_subsets],
            JOIN_SUBSETS_MERGE_HOW: self._join_subset_merge_how,
            MERGE_ON_FEATURES: self._merge_on_features,
            MERGE_HOW: self._merge_how,
            ALLOW_DUPLICATES: self._allow_duplicates,
            INPUT_APPEND_STR: self._input_append_str,
            JOIN_APPEND_STR: self._join_append_str,
            OUTPUT_FEATURE_FILE: self._output_features_file,
            OUTPUT_FILE: self._output_file,
        }
        config.update(super().get_config())
        return config

    def load_config(self, config: Dict) -> None:
        """
        Load the configuration.

        :param: config: Dictionary of configurations.
        :return: None
        """
        logging.debug(msg=f"Join.load_config(). config {config}.")
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

        # Configure the input data.
        self._input_data_files = config_list(config=config, key=INPUT_DATA_FILES, data_type=str, default_value=[])
        self._input_filters = config_filters(config=config, key=INPUT_FILTERS, default_value=[])
        for input_subset_value in config_list(config=config, key=INPUT_SUBSETS, data_type=Dict, default_value=[]):
            subset_filters = config_filters(config=input_subset_value, key='subset_filters', default_value=[])
            append_str = input_subset_value.get('subset_append_str', self._input_append_str)
            name = input_subset_value.get('subset_name', str(len(self._input_subsets)))
            self._input_subsets.append(JoinSubset(filters=subset_filters, append_str=append_str, name=name))
        self._input_subset_merge_how = config_string(config=config, key=INPUT_SUBSETS_MERGE_HOW,
                                                     default_value=self._input_subset_merge_how)

        # Configure the join data.
        self._join_features_file = config_string(config=config, key=JOIN_FEATURES_FILE, default_value=None)
        self._join_data_files = config_list(config=config, key=JOIN_DATA_FILES, data_type=str, default_value=[])
        self._join_filters = config_filters(config=config, key=JOIN_FILTERS, default_value=[])
        for join_subset_value in config_list(config=config, key=JOIN_SUBSETS, data_type=Dict, default_value=[]):
            subset_filters = config_filters(config=join_subset_value, key='subset_filters', default_value=[])
            append_str = join_subset_value.get('subset_append_str', self._join_append_str)
            name = join_subset_value.get('subset_name', str(len(self._join_subsets)))
            self._join_subsets.append(JoinSubset(filters=subset_filters, append_str=append_str, name=name))
        self._join_subset_merge_how = config_string(config=config, key=JOIN_SUBSETS_MERGE_HOW,
                                                    default_value=self._join_subset_merge_how)

        if config_list(config=config, key=JOIN_FEATURES_LIST, data_type=str, default_value=None):
            logging.warning(f"Deprecated parameter join_features_list. Use merge_on_features instead.")
            self._merge_on_features = config_list(config=config, key=JOIN_FEATURES_LIST, data_type=str,
                                                  default_value=None)
        self._merge_on_features = config_list(config=config, key=MERGE_ON_FEATURES, data_type=str, default_value=None)
        self._merge_how = config_string(config=config, key=MERGE_HOW, default_value=self._merge_how)
        self._allow_duplicates = config_boolean(config=config, key=ALLOW_DUPLICATES, default_value=None)
        self._input_append_str = config_string(config=config, key=INPUT_APPEND_STR,
                                               default_value=self._input_append_str)
        self._join_append_str = config_string(config=config, key=JOIN_APPEND_STR,
                                              default_value=self._join_append_str)
        self._output_features_file = config_string(config=config, key=OUTPUT_FEATURE_FILE, default_value=None)
        self._output_file = config_string(config=config, key=OUTPUT_FILE, default_value=None)
