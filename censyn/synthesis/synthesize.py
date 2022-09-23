import datetime
import json
import logging
import logging.handlers
import multiprocessing
import numbers
import os.path
from queue import Queue
from functools import partial
from typing import Any, Dict, List, Generator, Tuple, Union

import numpy as np
import pandas as pd

from censyn.checks import ConsistencyCheck, ParseError
from censyn.datasources import save_data_source, validate_data_source
from censyn.datasources.experiment_reader import ExperimentFile
from censyn.experiments import ExperimentGenerator
from censyn.experiments import Experiment
from censyn.features import Feature
from censyn.features import ModelChoice
from censyn.filters import ColumnEqualsFilter
from censyn.programs.censyn_base import CenSynBase
from censyn.report import Report
from censyn.results import ResultLevel
from censyn.results import Result, FloatResult, IndexResult, StrResult, MappingResult, ModelResult, TableResult
from censyn.utils import bounded_pool
from ..utils.config_util import config_boolean, config_string, config_list, config_filters
from ..utils.file_util import append_to_file_name
from ..utils.logger_util import get_logging_level, get_logging_file, listener_process, listener_configure

HEADER_TEXT: str = 'Synthesize Summary: \n' \
                   'The synthesize was run on {start_time} and took {duration} seconds. \n'

# Configuration labels
EXPERIMENT_FILE: str = 'experiment_file'
PROCESS_FEATURES: str = 'process_features'
BOOTSTRAP_FEATURES: str = 'bootstrap_features'
POST_PROCESS_FEATURES: str = 'post_process_features'
IGNORE_FEATURES: str = 'ignore_features'
WEIGHT_FEATURES: str = 'weight_features'
DATA_FILES: str = 'data_files'
FILTERS: str = 'filters'
EXTERNAL_BOOTSTRAP_DATA_FILES: str = 'external_bootstrap_data_files'
EXTERNAL_BOOTSTRAP_FILTERS: str = 'external_bootstrap_filters'
SYNTHESIZE_EXTERNAL_BOOTSTRAP_DATA: str = 'synthesize_external_bootstrap_data'
INDEPENDENT_FEATURE_NAME: str = 'independent_feature_name'
INDEPENDENT_FEATURE_FILTERS: str = 'independent_feature_filters'
SYNTHESIZE_FILTERS: str = 'synthesize_filters'
CONSISTENCY_CHECKS_FILE: str = 'consistency_checks_file'
MODEL_FILE: str = 'model_file'
OUTPUT_DATA_FILE: str = 'output_data_file'
ALL_CONFIG_ITEMS = [EXPERIMENT_FILE, PROCESS_FEATURES, BOOTSTRAP_FEATURES, POST_PROCESS_FEATURES,
                    IGNORE_FEATURES, WEIGHT_FEATURES, DATA_FILES, FILTERS,
                    EXTERNAL_BOOTSTRAP_DATA_FILES, EXTERNAL_BOOTSTRAP_FILTERS, SYNTHESIZE_EXTERNAL_BOOTSTRAP_DATA,
                    INDEPENDENT_FEATURE_NAME, INDEPENDENT_FEATURE_FILTERS, SYNTHESIZE_FILTERS,
                    CONSISTENCY_CHECKS_FILE, MODEL_FILE, OUTPUT_DATA_FILE]


class Synthesize(CenSynBase):
    def __init__(self, config_file: str) -> None:
        """
        Initialization of Synthesize.

        :param: config_file: Configuration file containing the parameters for synthesis.
        """
        super().__init__()
        logging.debug(msg=f"Instantiate Synthesize with config_file {config_file}.")

        self._experiment_file = None
        self._process_features = None
        self._post_process_features = []
        self._bootstrap_features = []
        self._ignore_features = []
        self._weight_features = []
        self._data_files = None
        self._data_df = pd.DataFrame()
        self._filters = []
        self._external_bootstrap_data_files = None
        self._external_synth_df = pd.DataFrame()
        self._external_synth_data_filters = []
        self._synthesize_external_bootstrap_data = False
        self._independent_feature_name = ""
        self._independent_feature_filters = []
        self._synthesize_filters = []
        self._consistency_checks_file = None
        self._consistency_checks = {}
        self._model_file = None
        self._output_data_file = None

        # Load the configuration data
        if config_file:
            with open(config_file, 'r') as w:
                config = json.loads(w.read())
                self.load_config(config)

        # Create report
        self.initialize_report()

        # Validate the settings
        if self._output_data_file:
            if not validate_data_source(self._output_data_file):
                msg = f"Must provide a supported output data file. Found {self._output_data_file} instead."
                logging.error(msg=msg)
                raise ValueError(msg)

        # Load the consistency checks
        if self._consistency_checks_file:
            with open(self._consistency_checks_file, 'r') as w:
                checks = json.loads(w.read())
                consistency_checks = checks.get('consistency_checks', [])
                for k, v in consistency_checks.items():
                    self._add_consistency_check(name=k, check=v)

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
        logging.debug(msg=f"Synthesize.create_features(). feature_def {feature_def} and features_name {features_name}.")
        if self._process_features:
            all_features = set(self._process_features + self._ignore_features + self._weight_features +
                               self._post_process_features)
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

    def execute(self) -> None:
        """Execute the synthesize process."""
        logging.debug(msg=f"Synthesize.execute().")
        e_start_time = datetime.datetime.now()

        self.load_features()
        self.validate_features()

        # Load the data
        self._data_df = self.report.time_function('Load-data-time',
                                                  self.load_data, {'file_names': self._data_files,
                                                                   'features': self.features})

        # Generate any data
        self._data_df = self.report.time_function('Generate-data-time',
                                                  self.generate_data, {'data_df': self._data_df,
                                                                       'features': self.features})

        # Filter the data
        self._data_df = self.report.time_function('Filter-data-time',
                                                  self.filter_data, {'filters': self._filters,
                                                                     'data_df': self._data_df})

        # external_bootstrap_data
        if self._external_bootstrap_data_files:
            self._external_synth_df = self.report.time_function('Load-data-time',  self.load_data,
                                                                {'file_names': self._external_bootstrap_data_files,
                                                                 'features': self.features,
                                                                 'data_name': "External Bootstrap Data"})
            # Generate any data
            self._external_synth_df = self.report.time_function('Generate-data-time', self.generate_data,
                                                                {'data_df': self._external_synth_df,
                                                                 'features': self.features})

            # Filter the data
            self._external_synth_df = self.report.time_function('Filter-data-time', self.filter_data,
                                                                {'filters': self._external_synth_data_filters,
                                                                 'data_df': self._external_synth_df,
                                                                 'data_name': "External Bootstrap Data"})

            self._validate_external_bootstrap_data()

        # Initialize the experiments
        experiments = self._experiments()

        # Process the experiments
        for e in experiments:
            if self._independent_feature_name:
                # Synthesize the data one value of the feature at a time.
                level = get_logging_level()
                syn_df = pd.DataFrame()
                ind_done = {f_code: False for f_code in self._data_df[self._independent_feature_name].unique()}

                def feature_code_gen() -> Generator:
                    """ Generator for processing an experiment for each code value of the independent feature."""
                    for ind_key, ind_flag in ind_done.items():
                        if not ind_flag:
                            yield partial(self._run_independent_experiment, ind_queue, level, experiment=e,
                                          feature_code=ind_key, processes=1)

                # Set up logging queue
                ind_queue = None
                ind_listener = None
                while not all(ind_done.values()):
                    if self.processes > 1:
                        # Set up multiprocessing logging queue
                        if ind_queue is None:
                            ind_queue = multiprocessing.Manager().Queue(-1)
                            ind_listener = multiprocessing.Process(target=listener_process,
                                                                   args=(ind_queue, listener_configure,
                                                                         get_logging_file()))
                            ind_listener.start()
                        ind_returns = bounded_pool(process_num=self.processes,
                                                   functions=feature_code_gen(),
                                                   process_factor=1)
                    else:
                        ind_returns = (task() for task in feature_code_gen())
                    if not ind_returns:
                        logging.error(f"Empty returns from run independent experiment")
                        break
                    for cur_value in ind_returns:
                        logging.info(f"Feature Value: {cur_value[0]} Merging results")
                        if isinstance(cur_value[0], numbers.Number):
                            if np.isnan(cur_value[0]):
                                for ind_k in ind_done.keys():
                                    if np.isnan(ind_k):
                                        ind_done[ind_k] = True
                                        break
                            else:
                                ind_done[cur_value[0]] = True
                        else:
                            ind_done[cur_value[0]] = True
                        syn_df = pd.concat([syn_df, cur_value[1]])
                        if cur_value[2]:
                            ind_results = {}
                            with open(cur_value[2], 'r') as report_file:
                                text = report_file.read()
                                metrics = text.split('Metric: ')
                                for metric_text in metrics:
                                    res = self.create_result(metric_text)
                                    if res:
                                        ind_results.update(res)
                                    elif metric_text.startswith("Model feature usage, Feature "):
                                        folder = cur_value[2][:-4]
                                        if os.path.isdir(folder):
                                            usage_name = metric_text[len("Model feature usage, "
                                                                         "Feature "):].split(' ')[0]
                                            usage_file = os.path.join(folder, f"Feature {usage_name} usage.csv")
                                            usage_df = pd.read_csv(usage_file, sep=',',)
                                            res = TableResult(value=usage_df, factor=usage_df.shape[0],
                                                              sort_column='Importance', ascending=False,
                                                              metric_name='Model feature usage',
                                                              description=f'Feature {usage_name} usage')
                                            ind_results.update({f"Model feature usage {usage_name}": res})
                                self._merge_report(from_results=ind_results, to_report=self.report)
                if ind_queue is not None:
                    ind_queue.put_nowait(None)
                    ind_listener.join()
            else:
                syn_df = self._run_experiment(experiment=e, data_df=self._data_df,
                                              external_synth_df=self._external_synth_df,
                                              ex_report=self.report, processes=self.processes)

            # Create cross dependencies results
            self._cross_dependencies()

            # Consistency checks on the synthesized data
            self._check_consistency(in_df=syn_df, to_report=self.report)
            # Save synthesized data file
            if self._output_data_file:
                save_data_source(file_name=self._output_data_file, in_df=syn_df)

        # Configuration result
        config_result = MappingResult(value=self.get_config(), level=ResultLevel.GENERAL,
                                      metric_name="Synthesize configuration")
        self.report.add_result("Synthesize configuration", config_result)

        # Create the output report
        logging.info('Create report')
        e_duration = datetime.datetime.now() - e_start_time
        self.report.header = HEADER_TEXT.format(start_time=e_start_time,
                                                duration=e_duration.total_seconds())
        self.report.produce_report()
        logging.info('Finish synthesize')

    def validate_features(self) -> None:
        """Check for valid dependencies for the features."""
        logging.debug(msg=f"Synthesize.validate_features().")
        super().validate_features(self.features)
        self.validate_filter_features(filters=self._filters, features=self.features)
        self.validate_filter_features(filters=self._external_synth_data_filters, features=self.features)
        self.validate_filter_features(filters=self._independent_feature_filters, features=self.features)
        self.validate_filter_features(filters=self._synthesize_filters, features=self.features)

        if self._process_features:
            self.validate_list_features(self._process_features, self.features, feature_name="Process Features")
        self.validate_list_features(self._post_process_features, self.features, feature_name="Post Process Features")
        self.validate_list_features(self._bootstrap_features, self.features, feature_name="Bootstrap Features")
        self.validate_list_features(self._ignore_features, self.features, feature_name="Ignore Features")
        self.validate_list_features(self._weight_features, self.features, feature_name="Weight Features")
        self.validate_list_features(self._process_features, self.features, feature_name="Process Features")
        if self._independent_feature_name:
            self.validate_list_features(feat_l=[self._independent_feature_name], features=self.features,
                                        feature_name="Independent Feature")

        self._validate_consistency_checks()

    def _cross_dependencies(self) -> None:
        """Create Cross Dependencies result."""
        logging.debug(msg=f"Synthesize._cross_dependencies().")
        if self.report is not None:
            cross_dep = {}
            for f_name in self.features.keys():
                feat_usage_result = self.report.results.get(f'Model feature usage Feature {f_name} usage', None)
                if feat_usage_result:
                    for index, row in feat_usage_result.value.iterrows():
                        cross_usage_key = f'Model feature usage Feature {row["Feature"]} usage'
                        cross_usage_res = self.report.results.get(cross_usage_key, None)
                        if cross_usage_res:
                            cross_df = cross_usage_res.value.loc[cross_usage_res.value["Feature"] == f_name]
                            if not cross_df.empty:
                                if cross_dep.get((row["Feature"], f_name), None):
                                    continue
                                cross_key = (f_name, row["Feature"])
                                cross_importance = row["Importance"] * cross_df["Importance"].sum()
                                cross_dep[cross_key] = cross_importance
            cross_dep = {k: v for k, v in sorted(cross_dep.items(), key=lambda item: item[1], reverse=True)}
            self.report.add_result(key="Cross Dependency", value=MappingResult(value=cross_dep,
                                                                               metric_name="Cross Dependency",
                                                                               description=""))

    def _validate_external_bootstrap_data(self) -> None:
        """Validate the external bootstrap data."""
        logging.debug(msg=f"Synthesize._validate_external_bootstrap_data().")
        required_f = set(self._bootstrap_features + self._ignore_features + self._weight_features)
        if self._synthesize_filters:
            required_f.update(self._process_features)
            required_f.update(self._post_process_features)

        for feat in self.features.values():
            model = feat.model_type.model
            if model == ModelChoice.NoopModel or model == ModelChoice.RandomModel:
                if feat.feature_name in required_f:
                    continue
                required_f.add(feat.feature_name)
        extra_f = [feat for feat in self._external_synth_df.columns if feat not in required_f]
        if len(extra_f) > 0:
            logging.warning(f"External Bootstrap Data has extra columns of {extra_f}.")
        missing_f = [feat for feat in required_f if feat not in self._external_synth_df.columns]
        if len(missing_f) > 0:
            logging.error(f"External Bootstrap Data has missing columns of {missing_f}.")
            raise ValueError(f"External Bootstrap Data has missing columns of {missing_f}.")

    def _validate_consistency_checks(self) -> None:
        """Check for valid consistence check expressions"""
        logging.debug(msg=f"Synthesize._validate_consistency_checks().")
        for k, v in self._consistency_checks.items():
            try:
                data_calculator = ConsistencyCheck(expression=v[0].expression)
                dependencies = data_calculator.compute_variables()
                for feat in dependencies:
                    if feat not in self.features:
                        msg = f"Feature {feat} of consistence check {k} with expression '{v[0].expression}' " \
                              f"does not exist."
                        logging.error(msg=msg)
                        raise ValueError(msg)
            except ParseError as e:
                logging.error(f"Grammar parse error on consistence check {k} with expression '{v[0].expression}'")
                raise ParseError(e)

    def _check_consistency(self, in_df: pd.DataFrame, to_report: Report) -> None:
        """
        Run consistency checks on the input DAtaFrame.

        :param: in_df: DataFrame to check.
        :param: to_report: The report to output the consistency metrics.
        :return: None
        """
        logging.debug(msg=f"Synthesize._check_consistency(). in_df {in_df.shape} and to_report {to_report}.")
        for k, v in self._consistency_checks.items():
            s = v[0].execute(in_df=in_df)
            if len(s) > 0:
                name = f'Consistency check {k}'
                to_report.add_result(key=name, value=IndexResult(value=s, metric_name=name, description=v[1]))

    @staticmethod
    def create_result(metric_text: str) -> Union[Dict, None]:
        """
        Create the result from another result text.

        :param: metric_text: Text of another result.
        :return: Dictionary of results.
        """
        logging.debug(msg=f"Synthesize.create_result(). metric_text {metric_text}.")
        float_names = ["encode-time", "train-time", "synthesize-time"]
        str_names = ["Synthesized features"]
        model_names = ["Model Description"]

        parts = [part.strip() for part in metric_text.split(':')]
        if len(parts) != 2:
            return None
        for name in float_names:
            if metric_text.startswith(name):
                description = parts[0][len(name) + 2:]
                return {name: FloatResult(value=float(parts[1]), metric_name=name, description=description)}
        for name in str_names:
            if metric_text.startswith(name):
                description = parts[0][len(name) + 2:]
                return {name: StrResult(value=parts[1], metric_name=name, description=description)}
        for name in model_names:
            if metric_text.startswith(name):
                value_d = {}
                description = parts[0][len(name) + 2:]
                if parts[1].startswith('Maximum depth='):
                    v_parts = [v_parts.strip() for v_parts in parts[1].split(',')]
                    if len(v_parts) != 2 or not v_parts[1].startswith('Number of leaves='):
                        raise ValueError(f'Unsupported ModelResult description {parts[1]}')
                    value_d['depth'] = int(v_parts[0][15:])
                    value_d['leaves'] = int(v_parts[1][18:])
                elif parts[1] == "Model was not trained because there was no data.":
                    value_d['depth'] = 0
                    value_d['leaves'] = 0
                elif parts[1] == "Model was trained.":
                    value_d['trained'] = True
                elif parts[1] == "Model was not trained.":
                    value_d['trained'] = False
                elif parts[1].startswith("Model dependencies "):
                    value_d['dependencies'] = parts[1][18:]
                else:
                    raise ValueError(f'Unsupported ModelResult description {parts[1]}')
                return {name + ' ' + description: ModelResult(value=value_d, metric_name=name,
                                                              description=description)}
        return None

    @staticmethod
    def _merge_report(from_results: Dict[str, Result], to_report: Report) -> None:
        logging.debug(msg=f"Synthesize._merge_report(). from_results {from_results} and to_report {to_report}.")
        for key, result in from_results.items():
            if isinstance(result, Result):
                r = to_report.results.get(key, None)
                if r is None:
                    to_report.add_result(key=key, value=result)
                elif isinstance(r, Result):
                    r.merge_result(result)

    def _experiments(self) -> List[Experiment]:
        """
        Creates the experiments from either an experiment json file or from the feature file.
        The experiment json file can be utilized to load a previously pickled experiment. If there is no
        experiment json file then an ExperimentGenerator uses the feature file to create the list of experiments.

        :return: List of the generated experiments.
        """
        logging.debug(msg=f"Synthesize._experiments().")
        # Load the experiments
        if self._experiment_file:
            logging.info(f'Loading experiment file {self._experiment_file}')
            return ExperimentFile(self._experiment_file).to_experiments()

        experiment_generator = ExperimentGenerator(feature_spec=self.features)
        return experiment_generator.generate()

    def _run_independent_experiment(self, log_q: Union[Queue, None], level: int,
                                    experiment: Experiment, feature_code: Any,
                                    processes: int) -> Tuple[Any, pd.DataFrame, str]:
        """
        Run an experiment on the independent feature with value of feature_code

        :param: log_q: Queue of logging handlers.
        :param: level: Logging level
        :param: experiment: The experiment to run.
        :param: feature_code: Value a for the independent features.
        :param: processes: The maximum number of parallel processes to utilize.
        :return: The synthesized DataFrame and Report
        """
        # The worker configuration is done at the start of the worker process run.
        # Note that on Windows you can't rely on fork semantics, so each process
        # will run the logging configuration code when it starts.
        root_logger = logging.getLogger()
        if not log_q and root_logger.hasHandlers():
            h = None
        else:
            h = logging.handlers.QueueHandler(log_q)  # Just the one handler needed
            root_logger.setLevel(level)
            root_logger.addHandler(h)

        feature_start_time = datetime.datetime.now()
        # Filter the data for the independent feature value
        ind_filter = ColumnEqualsFilter(header=self._independent_feature_name, value=feature_code)
        feature_df = ind_filter.execute(df=self._data_df)
        external_synth_df = ind_filter.execute(df=self._external_synth_df)
        logging.info(f"\nFeature Value: {feature_code} | SIZE: {feature_df.shape[0]} Records >> Synthesizing...")

        # Filter the data
        feature_df = self.filter_data(self._independent_feature_filters, feature_df,
                                      data_name="Independent Feature Data")
        external_synth_df = self.filter_data(self._independent_feature_filters, external_synth_df,
                                             data_name="External Bootstrap Data")

        append_fn = f'_{self._independent_feature_name}_{str(feature_code)}'
        report_file_path = self.report.full_file_path
        if report_file_path:
            file_path, file_name = os.path.split(report_file_path)
            root_name, ext = os.path.splitext(file_name)
            report_file_path = os.path.join(file_path, root_name)
        report = self.create_report(file_path=report_file_path, file_append=append_fn)
        ind_df = self._run_experiment(experiment=experiment, data_df=feature_df, external_synth_df=external_synth_df,
                                      ex_report=report, processes=processes, file_append=append_fn)
        if self._output_data_file:
            file_name = append_to_file_name(file_name=self._output_data_file, file_append=append_fn)
            save_data_source(file_name=file_name, in_df=ind_df)

        # Create the output report
        logging.info('Create report')
        feature_duration = datetime.datetime.now() - feature_start_time
        report.header = HEADER_TEXT.format(start_time=feature_start_time, duration=feature_duration.total_seconds())
        report.produce_report()

        root_logger.removeHandler(h)
        return feature_code, ind_df, report.full_file_path

    def _run_experiment(self, experiment: Experiment, data_df: pd.DataFrame, external_synth_df: pd.DataFrame,
                        ex_report: Report, processes: int, file_append: str = "") -> pd.DataFrame:
        """
        Run an experiment with the data.

        :param: experiment: The experiment to run.
        :param: data_df: Input data.
        :param: external_synth_df: DataFrame is the unencoded data frame that can use for the bootstrap
            of the synthesis process.
        :param: ex_report: Experiment reports.
        :param: processes: The maximum number of parallel processes to utilize.
        :param: file_append: String to append for written file names.
        :return: The synthesized DataFrame.
        """
        logging.debug(msg=f"Synthesize._run_experiment().")
        # Filter the data
        no_df = None
        if self._synthesize_filters:
            syn_df = self.filter_data(filters=self._synthesize_filters, data_df=data_df, data_name="Data")
            if syn_df.shape[0] < data_df.shape[0]:
                no_df = data_df.drop(syn_df.index)
                data_df = syn_df
            if not external_synth_df.empty:
                no_df = None
                syn_df = self.filter_data(filters=self._synthesize_filters, data_df=external_synth_df,
                                          data_name="External Bootstrap Data")
                if syn_df.shape[0] < external_synth_df.shape[0]:
                    no_df = external_synth_df.drop(syn_df.index)
                    external_synth_df = syn_df

        if data_df.empty:
            ex_df = data_df.copy()
        else:
            experiment.process_features = self._process_features
            experiment.bootstrap_features = self._bootstrap_features
            experiment.post_process_features = self._post_process_features
            experiment.ignore_features = self._ignore_features
            experiment.weight_features = self._weight_features
            experiment.synthesize_external_bootstrap_data = self._synthesize_external_bootstrap_data
            ex_df = ex_report.time_function('Experiment-time', experiment.run,
                                            {'data_df': data_df, 'external_synth_df': external_synth_df,
                                             'report': ex_report, 'processes': processes})
            if self._model_file:
                file_name = append_to_file_name(file_name=self._model_file, file_append=file_append)
                experiment.pickle_experiment_models(file_path=file_name)

        if no_df is not None:
            ex_df = pd.concat([ex_df, no_df], ignore_index=False)
            ex_df.sort_index(inplace=True)
        logging.debug(ex_df)

        return ex_df

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        :return: configuration dictionary
        """
        logging.debug(msg=f"Synthesize.get_config().")

        config = {
            EXPERIMENT_FILE: self._experiment_file,
            PROCESS_FEATURES: self._process_features,
            BOOTSTRAP_FEATURES: self._bootstrap_features,
            POST_PROCESS_FEATURES: self._post_process_features,
            IGNORE_FEATURES: self._ignore_features,
            WEIGHT_FEATURES: self._weight_features,
            DATA_FILES: self._data_files,
            FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()} for f in self._filters],
            EXTERNAL_BOOTSTRAP_DATA_FILES: self._external_bootstrap_data_files,
            EXTERNAL_BOOTSTRAP_FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                                         for f in self._external_synth_data_filters],
            SYNTHESIZE_EXTERNAL_BOOTSTRAP_DATA: self._synthesize_external_bootstrap_data,
            INDEPENDENT_FEATURE_NAME: self._independent_feature_name,
            INDEPENDENT_FEATURE_FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                                          for f in self._independent_feature_filters],
            SYNTHESIZE_FILTERS: [{'class': f.__class__.__name__, 'attributes': f.to_dict()}
                                 for f in self._synthesize_filters],
            CONSISTENCY_CHECKS_FILE: self._consistency_checks_file,
            MODEL_FILE: self._model_file,
            OUTPUT_DATA_FILE: self._output_data_file,
        }
        config.update(super().get_config())
        return config

    def save_config(self, config_file: str) -> None:
        """
        DEPRECATED
        Save a configuration file.

        :param: config_file: File name to save the configuration.
        :return: None
        """
        if config_file:
            with open(config_file, 'w') as outfile:
                json.dump(self.get_config(), outfile, indent=4)

    def load_config(self, config: Dict) -> None:
        """
        Load the configuration. This is a json dictionary loaded from a file.

        :param: config: Dictionary of the configuration.
        :return: None
        """
        logging.debug(msg=f"Synthesize.load_config(). config {config}.")
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

        self._experiment_file = config_string(config=config, key=EXPERIMENT_FILE, default_value=None)
        self._process_features = config_list(config=config, key=PROCESS_FEATURES, data_type=str, default_value=[])
        self._bootstrap_features = config_list(config=config, key=BOOTSTRAP_FEATURES, data_type=str, default_value=[])
        self._post_process_features = config_list(config=config, key=POST_PROCESS_FEATURES, data_type=str, default_value=[])
        self._ignore_features = config_list(config=config, key=IGNORE_FEATURES, data_type=str, default_value=[])
        self._weight_features = config_list(config=config, key=WEIGHT_FEATURES, data_type=str, default_value=[])
        self._data_files = config_list(config=config, key=DATA_FILES, data_type=str, default_value=[])
        self._filters = config_filters(config=config, key=FILTERS, default_value=[])
        self._external_bootstrap_data_files = config_list(config=config, key=EXTERNAL_BOOTSTRAP_DATA_FILES, data_type=str,
                                                          default_value=[])
        self._external_synth_data_filters = config_filters(config=config, key=EXTERNAL_BOOTSTRAP_FILTERS,
                                                           default_value=[])
        self._synthesize_external_bootstrap_data = config_boolean(config=config,
                                                                  key=SYNTHESIZE_EXTERNAL_BOOTSTRAP_DATA,
                                                                  default_value=None)
        self._independent_feature_name = config_string(config=config, key=INDEPENDENT_FEATURE_NAME,
                                                       default_value=None)
        self._independent_feature_filters = config_filters(config=config, key=INDEPENDENT_FEATURE_FILTERS,
                                                           default_value=[])
        self._synthesize_filters = config_filters(config=config, key=SYNTHESIZE_FILTERS, default_value=[])
        self._consistency_checks_file = config_string(config=config, key=CONSISTENCY_CHECKS_FILE,
                                                      default_value=None)
        self._model_file = config_string(config=config, key=MODEL_FILE, default_value=None)
        self._output_data_file = config_string(config=config, key=OUTPUT_DATA_FILE, default_value=None)

    def _add_consistency_check(self, name: str, check: Dict) -> None:
        logging.debug(msg=f"Synthesize._add_consistency_check(). name {name} and check {check}")
        class_name = check.get('class')
        attributes = check['attributes']
        if class_name == 'ConsistencyCheck':
            expr = attributes.get('expression', "")
            if isinstance(expr, List):
                expr = " ".join(expr)
            cur_check = ConsistencyCheck(expression=expr)
        else:
            msg = f"Must provide a supported ConsistencyCheck. Found {class_name} instead."
            logging.error(msg=msg)
            raise ValueError(msg)
        description = check.get('description', cur_check.to_dict())
        self._consistency_checks[name] = (cur_check, description)
