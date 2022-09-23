import dill
import logging
import logging.handlers
import multiprocessing
import math
import random
from itertools import chain
from functools import partial
from queue import Queue
from typing import List, Dict, Union

import pandas as pd
import psutil
import numpy as np

from censyn.encoder import EncodeProcessor
from censyn.features import Feature
from censyn.report import Report
from censyn.results import StrResult
from censyn.utils import pool_helper, bounded_pool, calculate_weights
from ..utils.logger_util import get_logging_level, get_logging_file, listener_process, listener_configure


class Experiment:

    def __init__(self, experiment_name: str, experiment_models: Dict, is_encoded: bool = False):
        """
        Initialization for running an experiment.

        :param: experiment_name: The name of the experiment.
        :param: experiment_models: The models to train.
        :param: is_encoded: If the data is already encoded. Default: False
        """
        logging.debug(msg=f"Instantiate Experiment with experiment_name {experiment_name}, "
                          f"experiment_models {experiment_models} and is_encoded {is_encoded}.")
        if experiment_models is None:
            raise ValueError('An experiment must have an experiment_models.')

        # Initial values
        self._data: Union[None, pd.DataFrame] = None
        self._data_unencoded: Union[None, pd.DataFrame] = None
        self._experiment_models: Dict = experiment_models
        self._experiment_name: str = experiment_name
        self._process_features: List[str] = [model.target_feature.feature_name for model in experiment_models.values()]
        self._bootstrap_features: List[str] = []
        self._post_process_features: List[str] = []
        self._synthesized_features: List[str] = []
        self._ignore_features: List[str] = []
        self._weight_features: List[str] = []
        self._synthesize_external_bootstrap_data = False
        self._is_encoded: bool = is_encoded
        self._encode_names = {}
        self._is_trained: bool = False

        # Set during run
        self._processes = 1

    def __str__(self) -> str:
        return f'Experiment Name: {self._experiment_name}'

    def encode(self, report: Report = None) -> None:
        """
        Encodes the data_unencoded based on each feature's feature_specs.

        :param: report: Report for the experiment.
        """
        logging.debug(msg=f"Experiment.encode(). report {report}.")
        if self._is_encoded:
            return

        features: List = [model.target_feature for model in self._experiment_models.values()]
        if report is not None:
            self._data = report.time_function('encode-time', self._encode_column_subset,
                                              {'df': self._data_unencoded.copy(), 'features': features})
        else:
            self._data = self._encode_column_subset(df=self._data_unencoded.copy(), features=features)

        self._is_encoded = True

    @staticmethod
    def _encode_column_subset(df: pd.DataFrame, features: List[Feature]) -> pd.DataFrame:
        """
        Given a list of features and a dataframe encode them and return the result.

        :param: df: the data frame that you want to encode.
        :param: features: a list of features that include encoders that will be used to encode the data frame.
        :return: Encoded data frame.
        """
        logging.debug(msg=f"Experiment._encode_column_subset(). df {df.shape} and features{features}.")
        logging.info(f'Start Encoding the data {df.shape[0]} rows by  {df.shape[1]} columns')
        processor = EncodeProcessor(inplace=True)
        for feature in features:
            processor.append_encoder(feature.encoder)
        processor.execute(in_df=df)
        logging.debug(f'Encoded Data size {df.shape[0]} rows by  {df.shape[1]} columns')
        return df

    def train(self, report: Report = None) -> None:
        """
        Performs training, timing it if the class has been provided with a report.

        :param: report: Report for the experiment.
        """
        logging.debug(msg=f"Experiment.train(). report {report}.")
        self.encode(report=report)

        if report is not None:
            report.time_function('train-time', self._train, {})
        else:
            self._train()

    def _train(self) -> None:
        """
        Performs the training for all the experiment models.
        Training is performed in parallel if processes > 1. The multiprocessing may not return all models if there
        was an exception such as: 'OverflowError: cannot serialize a byte object larger than 4 Gib.'.
        """
        logging.debug(msg=f"Experiment._train().")
        if self._is_trained:
            return

        logging.info('Start Training Models')
        num_process = 1
        if self._processes > 1:
            sys_mem = psutil.virtual_memory().available
            data_mem = np.sum(self.data.memory_usage(deep=True)) * 3.0
            fit_mem = max(math.floor(sys_mem / data_mem), 1)
            num_process = min(self._processes, fit_mem)
        logging.info(f'Training models on {num_process} process.')

        # Train the features with the bootstrap features.
        bootstrap_2 = self._train_models(bootstrap_f=self.bootstrap_features, num_process=num_process)

        # Train the original bootstrap features with the previous trained features.
        self._train_models(bootstrap_f=bootstrap_2, num_process=num_process)

        self._is_trained = True

    def _train_models(self, bootstrap_f: List[str], num_process: int = 1) -> List[str]:
        """
        Performs the training for the experiment models.
        
        :param: bootstrap_f: Feature names to use as bootstrap feature for training.
        :param: num_process: Number of processes to utilize.
        :return: List of trained feature names.
        """
        logging.debug(msg=f"Experiment._train(). bootstrap_f {bootstrap_f} and num_process {num_process}.")
        self.verify_dependencies(bootstrap_f)

        level = get_logging_level()
        trained_f = []
        if num_process > 1:
            train_queue = multiprocessing.Manager().Queue(-1)
            train_listener = multiprocessing.Process(target=listener_process,
                                                     args=(train_queue, listener_configure, get_logging_file()))
            train_listener.start()

            models = bounded_pool(num_process, self._train_model_gen(log_q=train_queue, level=level,
                                                                     bootstrap_f=bootstrap_f),
                                  process_factor=1)
            for model in models:
                trained_f.append(model.target_feature.feature_name)
                self._experiment_models[model.target_feature.feature_name] = model

            train_queue.put_nowait(None)
            train_listener.join()
            logging.debug(f'Train remaining models')

        # Train remaining models.
        models = (task() for task in self._train_model_gen(log_q=None, level=level, bootstrap_f=bootstrap_f))
        for model in models:
            trained_f.append(model.target_feature.feature_name)
            self._experiment_models[model.target_feature.feature_name] = model
        return trained_f

    def verify_dependencies(self, bootstrap_f: List[str]) -> None:
        """
        Verify the dependencies for the model features. Auto create the dependency list for trainable features which
        have an empty dependency list.

        :param: bootstrap_f: List of bootstrap feature names.
        :return: None.
        """
        logging.debug(msg=f"Experiment.verify_dependencies(). bootstrap_f {bootstrap_f}.")
        for model in self._experiment_models.values():
            model.calculate_dependencies()

        dependencies = bootstrap_f.copy()
        for f_name in self.process_features:
            model = self._experiment_models[f_name]
            if f_name not in bootstrap_f and f_name not in self._ignore_features:
                if model.needs_dependencies and not model.trained:
                    if len(model.target_feature.train_dependencies) == 0:
                        # Auto set dependencies for feature
                        cur_depends = [d for d in dependencies if d not in model.target_feature.exclude_dependencies]
                        model.target_feature.train_dependencies = cur_depends
                    else:
                        for o_name in self._verify_feature_dependencies(model.target_feature, dependencies, []):
                            # Need to set dependency to prevent circular dependency.
                            o_model = self._experiment_models[o_name]
                            if o_model.needs_dependencies and not o_model.trained and \
                                    len(o_model.target_feature.train_dependencies) == 0:
                                cur_depends = [d for d in dependencies if d not in
                                               model.target_feature.exclude_dependencies]
                                o_model.target_feature.train_dependencies = cur_depends
                            dependencies.append(o_name)
                dependencies.append(f_name)
        logging.debug(f'Finish verify_dependencies({bootstrap_f})')

    def _verify_feature_dependencies(self, feature: Feature, valid_f: List[str], restrict_f: List[str]) -> List[str]:
        """
        Verify the dependencies of the feature. Iterate through the dependencies' dependencies. If circular
        dependencies then ValueError will be raised.

        :param: feature: Feature to validate.
        :param: valid_f: Valid features for dependencies.
        :param: restrict_f: Restricted features for dependencies.
        :return: List of dependent feature names which do not have defined dependencies.
        """
        logging.debug(msg=f"Experiment._verify_feature_dependencies(). feature {feature},"
                          f"valid_f {valid_f} and restrict_f {restrict_f}.")
        invalid = set(feature.train_dependencies) & set(restrict_f)
        if len(invalid) > 0:
            raise ValueError(f"Circular dependency on feature {feature.feature_name} with {invalid}.")

        others = set(feature.train_dependencies) - set(valid_f) - set(self._ignore_features)
        if len(others) == 0:
            return []
        return_f = []
        cur_restrict = restrict_f.copy()
        cur_restrict.append(feature.feature_name)
        for o_name in others:
            if o_name not in return_f:
                return_f.append(o_name)
            if len(self._experiment_models[o_name].target_feature.train_dependencies) > 0:
                extra_f = self._verify_feature_dependencies(feature=self._experiment_models[o_name].target_feature,
                                                            valid_f=valid_f,
                                                            restrict_f=cur_restrict)
                for feat in extra_f:
                    if feat not in return_f:
                        return_f.append(feat)
        return return_f

    def _train_model_gen(self, log_q: Union[Queue, None], level: int, bootstrap_f: List[str]):
        """
        Generator function for the training of the models.

        :param: log_q: Queue of logging handlers.
        :param: level: Logging level
        :param: bootstrap_f: List of bootstrap feature names.
        :return: Generator function for training single model
        """
        logging.debug(msg=f"Experiment._train_model_gen(). bootstrap_f {bootstrap_f}.")
        # Generate child seed sequences
        random_state = np.random.get_state()
        child_seeds = iter(np.random.SeedSequence(entropy=random_state[1]).spawn(len(self._experiment_models)))

        # Weight data
        weight_data = calculate_weights(in_df=self.data_unencoded, features=self.weight_features)

        for f_name in self.process_features:
            model = self._experiment_models[f_name]
            if f_name not in bootstrap_f and f_name not in self._ignore_features:
                assert not (not model.needs_dependencies and len(model.target_feature.train_dependencies) > 0), \
                    f'Model {f_name} not needs_dependencies with train_dependencies.'
                assert not (model.needs_dependencies and len(model.target_feature.train_dependencies) == 0), \
                    f'Model {f_name} needs_dependencies without train_dependencies.'
                if not model.trained:
                    all_dependencies = [self.experiment_models[x].target_feature.encoder.indicator_and_encode_names
                                        for x in model.target_feature.train_dependencies]
                    training = self.data[chain.from_iterable(all_dependencies)].copy()
                    target = self.data_unencoded[f_name].copy()
                    yield partial(self._train_single_model, log_q=log_q, level=level, model=model,
                                  training_data=training, target_data=target,
                                  weight=weight_data.copy() if weight_data is not None else None,
                                  seed=next(child_seeds))

    @staticmethod
    def _train_single_model(log_q: Union[Queue, None], level: int,
                            model, training_data: pd.DataFrame, target_data: pd.Series,
                            weight: pd.Series, seed: np.random.SeedSequence = None):
        """
        Train a single model using the training data to the target data.

        :param: log_q: Queue of logging handlers.
        :param: level: Logging level
        :param: model: Object of type Model (Can't be typed in the function
        signature because of the structure of inheritance)
        :param: training_data: Data to train the model with.
        :param: target_data: Labels to classify the training_data.
        :param: weight: Weight of the samples.
        :param: seed: Numpy random SeedSequence entry for the process.
        :return: Trained model
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

        if seed:
            np.random.seed(seed.entropy)

        logging.info(f'Training model {model.target_feature.feature_name} with data size {training_data.shape}')
        model.train(predictor_df=training_data, target_series=target_data, weight=weight)

        # explicitly free the memory
        del training_data  # delete copy of training data since it is a copy and not needed any more.
        root_logger.removeHandler(h)
        return model

    def _dig(self, models: Dict, syn_df: pd.DataFrame, encode_syn_df: pd.DataFrame, encode_names: Dict[str, List[str]],
             seed: np.random.SeedSequence = None) -> tuple:
        """
        Synthesize data for each feature in the experiment models. This method synthesize
        exactly the number of records presents in the prediction data.

        :param: models: The experiment Keys: Features, Values: Models
        :param: syn_df: Synthesized DataFrame
        :param: encode_syn_df: Encoded synthesized DataFrame
        :param: encode_names: Dictionary of the encode feature names
        :return: The unencoded and encoded synthesize data
        """
        logging.debug(msg=f"Experiment._dig(). models {models}, syn_df {syn_df.shape}, "
                          f"encode_syn_df {encode_syn_df} and encode_names {encode_names}.")

        def _dig_feature(feature: Feature) -> None:
            """
            Given a feature, descend the dependencies tree and synthesize from the leaves up.

            :param: feature: The feature to be synthesized.
            """
            nonlocal syn_df
            nonlocal encode_syn_df
            nonlocal encode_names
            # If we have already synthesized this data don't retry
            if {feature.feature_name}.issubset(syn_df.columns):
                return

            logging.debug(f'Synthesize feature {feature.feature_name}.')
            if feature.train_dependencies is not None and len(feature.train_dependencies) != 0:
                for depend in feature.train_dependencies:
                    _dig_feature(models[depend].target_feature)
                if not all(f in syn_df.columns for f in feature.train_dependencies):
                    raise ValueError(f'The dependent features for {feature.feature_name} are not in the data')
                if models[feature.feature_name].encode_dependencies:
                    dependent_enc_names = list(chain.from_iterable([encode_names[f_depends]
                                                                    for f_depends in feature.train_dependencies]))
                    # Get encoded synthesized dependent features data from memorized encoded synth data.
                    df_to_send = encode_syn_df[dependent_enc_names].copy()
                else:
                    df_to_send = syn_df[feature.train_dependencies].copy()
            else:
                df_to_send = None
            # Create the synthesize feature data
            logging.info(f'Synthesize model {feature.feature_name}')
            series = models[feature.feature_name].synthesize(input_data=df_to_send)

            syn_df = pd.concat([syn_df, series.to_frame(name=feature.feature_name)], axis=1)

            # Encode synthesized data for the feature in context
            enc_feature_data = syn_df[[feature.feature_name]].copy()
            feature.encoder.encode(enc_feature_data)

            # Memorize feature's encoded synthesized data by appending it to the enc_synth_d dataframe and
            # also store the encoded feature names as a mapping of k: feature_name V: enc feature_names
            encode_names[feature.feature_name] = [c for c in enc_feature_data.columns]
            encode_syn_df = pd.concat([encode_syn_df, enc_feature_data], axis=1)
            synthesized_features.append(feature.feature_name)
            logging.debug(f'Finish synthesize feature {feature.feature_name}.')

        # Set any required random seed
        if seed:
            np.random.seed(seed.entropy)
            random.seed(np.random.random_integers(2**31 - 1))

        for model in models.values():
            synthesized_features: List[str] = []
            _dig_feature(feature=model.target_feature)
            self._synthesized_features.extend(synthesized_features)

        return syn_df, encode_syn_df

    def synthesize(self, external_synth_df: pd.DataFrame, report: Report = None) -> pd.DataFrame:
        """
        Performs synthesis, timing the process if a report has been provided to the class.

        :param: external_synth_df: DataFrame of unencoded data that is used for the bootstrap of the synthesis.
        :param: report: Report for the experiment.
        :return: pd.DataFrame - The synthesized data set.
        """
        logging.debug(msg=f"Experiment.synthesize(). external_synth_df {external_synth_df.shape} and "
                          f"report {report}.")
        logging.info('Start synthesize')
        if report is not None:
            return report.time_function('synthesize-time', self._synthesize,
                                        {'external_synth_df': external_synth_df})
        return self._synthesize(external_synth_df)

    def _synthesize(self, external_synth_df: pd.DataFrame, seed: np.random.SeedSequence = None) -> pd.DataFrame:
        """Synthesizes data from the trained Models in the experiment.

        :param: external_synth_df: DataFrame of unencoded data that is used for the bootstrap of the synthesis.
        :return: pd.DataFrame - The synthesized data set.
        """
        logging.debug(msg=f"Experiment._synthesize(). external_synth_df {external_synth_df.shape}.")
        if not self._is_trained:
            raise ValueError('This experiment is not trained.')

        # Initialize with the synthesized features.
        self._synthesized_features = []

        # Create initial Synthesize Data
        if not external_synth_df.empty:
            if not all(f in external_synth_df.columns for f in self.bootstrap_features):
                raise ValueError(f"Bootstrap features {self.bootstrap_features} not in {external_synth_df.columns}")
            if not all(f in external_synth_df.columns for f in self.ignore_features):
                raise ValueError(f"Ignore features {self.ignore_features} not in {external_synth_df.columns}")

            # Re-train model that do not have dependencies such as NoopModels.
            for model in self._experiment_models.values():
                if model.trained and not model.needs_dependencies:
                    if model.target_feature.feature_name not in external_synth_df.columns:
                        raise ValueError(f"Feature {model.target_feature.feature_name} with Model "
                                         f"{model.target_feature.model_type.model.name} is not in external data "
                                         f"{external_synth_df.columns}")
                    target_s = external_synth_df[model.target_feature.feature_name]
                    model.train(predictor_df=None, target_series=target_s, weight=None)
            syn_df = pd.DataFrame(data=external_synth_df, columns=set(self.bootstrap_features + self.ignore_features))
        else:
            syn_df = pd.DataFrame(data=self.data_unencoded, columns=set(self.bootstrap_features + self.ignore_features))
        encode_syn_df = syn_df.copy()
        features: List = [model.target_feature for model in self._experiment_models.values()
                          if model.target_feature.feature_name in encode_syn_df.columns]
        self._encode_column_subset(df=encode_syn_df, features=features)

        # Synthesize the features
        syn_df, encode_syn_df = self._synthesize_features(syn_df=syn_df, encode_syn_df=encode_syn_df)

        # Clear the bootstrap features from the synthesize-data
        for f_name in self.bootstrap_features:
            if f_name in self.ignore_features:
                continue
            if f_name in external_synth_df.columns and not self._synthesize_external_bootstrap_data:
                continue
            syn_df.drop(labels=f_name, axis=1, inplace=True)
            encode_labels = self.experiment_models[f_name].target_feature.encoder.indicator_and_encode_names
            encode_syn_df.drop(labels=encode_labels, axis=1, inplace=True)

        # Synthesize the original bootstrap features
        syn_df, encode_syn_df = self._synthesize_features(syn_df=syn_df, encode_syn_df=encode_syn_df)

        # Clear the post process features from the synthesize-data
        for f_name in self.post_process_features:
            if f_name not in external_synth_df.columns and f_name in syn_df.columns:
                syn_df.drop(labels=f_name, axis=1, inplace=True)
                encode_labels = self.experiment_models[f_name].target_feature.encoder.indicator_and_encode_names
                encode_syn_df.drop(labels=encode_labels, axis=1, inplace=True)

        # Synthesize the post process features
        syn_df, encode_syn_df = self._synthesize_features(syn_df=syn_df, encode_syn_df=encode_syn_df)
        return syn_df

    def _synthesize_features(self, syn_df: pd.DataFrame, encode_syn_df: pd.DataFrame) -> tuple:
        """
        Synthesize data for each feature in the experiment models.

        :param: syn_df: Synthesized DataFrame
        :param: encode_syn_df: Encoded synthesized DataFrame
        :return: The unencoded and encoded synthesize data
        """
        logging.debug(msg=f"Experiment._synthesize_features(). syn_df {syn_df.shape} and "
                          f"encode_syn_df {encode_syn_df}.")
        encode_names: Dict[str, List[str]] = {
            f_name: self.experiment_models[f_name].target_feature.encoder.indicator_and_encode_names
            for f_name in syn_df.columns}

        # Calculate number of slices of the data for processing.
        num_process = 1
        if self._processes > 1:
            data_mem = np.sum(self.data.memory_usage(deep=True))
            un_encode_data_mem = np.sum(self.data_unencoded.memory_usage(deep=True))
            fit_mem = math.ceil((data_mem + un_encode_data_mem) / (512 * 1024 * 1024))
            if fit_mem > 1:
                # TODO Remove setting number of process to 1 when can successfully synthesize on multiple processes.
                # num_process = max(self._processes, fit_mem)
                num_process = 1
        logging.debug(f'Synthesize_features number of process is {num_process}')

        if num_process > 1:
            # Generate child seed sequences and tasks
            random_state = np.random.get_state()
            child_seeds = iter(np.random.SeedSequence(entropy=random_state[1]).spawn(num_process))
            tasks = [partial(self._dig, models=self._experiment_models, syn_df=syn_df.loc[chunk.index, :],
                             encode_syn_df=encode_syn_df.loc[chunk.index, :], encode_names=encode_names,
                             seed=next(child_seeds))
                     for chunk in np.array_split(self._data, num_process)]

            try:
                syn_chunks = pool_helper(self._processes, tasks)
                syn_df = pd.concat(s_df for s_df, e_df in syn_chunks)
                encode_syn_df = pd.concat(e_df for s_df, e_df in syn_chunks)
            except OverflowError:
                logging.debug(f'OverflowError in synthesize_feature.')
                syn_df, encode_syn_df = self._dig(models=self._experiment_models, syn_df=syn_df,
                                                  encode_syn_df=encode_syn_df, encode_names=encode_names)
        else:
            syn_df, encode_syn_df = self._dig(models=self._experiment_models, syn_df=syn_df,
                                              encode_syn_df=encode_syn_df, encode_names=encode_names)
        return syn_df, encode_syn_df

    def run(self, data_df: pd.DataFrame, external_synth_df: pd.DataFrame,
            report: Report = None, processes: int = 1) -> pd.DataFrame:
        """
        This function is shorthand for running both train and synthesize. It will generate reports.
        Returns the synthesized data.

        :param: data_df {pd.DataFrame} -- This is the unencoded data frame that you want to process.
        :param: external_synth_df: DataFrame of unencoded data that is used for the bootstrap of the synthesis.
        :param: report: Report for the experiment.
        :param: processes: The maximum number of parallel processes to utilize. Default = 1.
        :return: pd.DataFrame - The synthesized data set.
        """
        logging.debug(msg=f"Experiment.run(). data_df {data_df.shape}, "
                          f"external_synth_df {external_synth_df.shape}, report {report} and processes {processes}.")
        # Initialize the data
        self._data_unencoded = data_df
        self._data = None
        self._is_encoded = False
        self._processes = processes
        self._is_trained = False
        for model in self._experiment_models.values():
            model.clear()

        # Check that weight features are in the ignore feature list.
        for feat in self.weight_features:
            if feat not in self.ignore_features:
                self.ignore_features.append(feat)

        self.train(report)
        syn_df = self.synthesize(external_synth_df, report)

        if report is not None:
            syn_report = StrResult(value=self._synthesized_features, metric_name="Synthesized features",
                                   description="The order of the synthesized features")
            report.add_result(key=syn_report.metric_name, value=syn_report)
            for model in self._experiment_models.values():
                for res in model.get_results(encode_names=self.encode_names):
                    report.add_result(key=f"{res.metric_name} {res.description}", value=res)

        # Set order of columns the same as the input order with additional columns appended.
        columns = [c for c in self._data_unencoded.columns if c in syn_df.columns]
        return syn_df.reindex(columns=columns)

    def pickle_experiment_models(self, file_path: str) -> None:
        """
        Save the experiment models in pickle format.

        :param: file_path: String of the file name to save.
        :return: None
        """
        logging.debug(msg=f"Experiment.pickle_experiment_models(). file_path {file_path}.")
        with open(file_path, 'wb') as dill_file:
            dill.dump(self._experiment_models, dill_file, protocol=-1)

    @property
    def encode_names(self) -> Dict:
        """Feature encode names getter"""
        logging.debug(msg=f"Experiment.encode_names().")
        if len(self._encode_names) == 0:
            encode_names = {}
            for model in self._experiment_models.values():
                encode_names.update({encode_n: model.target_feature.feature_name
                                     for encode_n in model.target_feature.encoder.indicator_and_encode_names})
            self._encode_names = encode_names
        return self._encode_names

    @property
    def data(self) -> pd.DataFrame:
        """Data getter"""
        if self._data is None:
            self._data = self._data_unencoded.copy()

        return self._data

    @property
    def data_unencoded(self) -> pd.DataFrame:
        """data_unencoded getter"""
        return self._data_unencoded

    @property
    def experiment_models(self) -> Dict:
        """experiment_models getter"""
        return self._experiment_models

    @property
    def experiment_name(self) -> str:
        """experiment_name getter"""
        return self._experiment_name

    @property
    def process_features(self) -> List[str]:
        """Training process features' names"""
        return self._process_features

    @process_features.setter
    def process_features(self, value: List[str]):
        self._process_features = value

    @property
    def bootstrap_features(self) -> List[str]:
        """Training bootstrap features' names"""
        return self._bootstrap_features

    @bootstrap_features.setter
    def bootstrap_features(self, value: List[str]):
        self._bootstrap_features = value

    @property
    def post_process_features(self) -> List[str]:
        """Training post process features' names"""
        return self._post_process_features

    @post_process_features.setter
    def post_process_features(self, value: List[str]):
        self._post_process_features = value

    @property
    def ignore_features(self) -> List[str]:
        """Training ignore features' names"""
        return self._ignore_features

    @ignore_features.setter
    def ignore_features(self, value: List[str]):
        self._ignore_features = value

    @property
    def weight_features(self) -> List[str]:
        """Training weight features' names"""
        return self._weight_features

    @weight_features.setter
    def weight_features(self, value: List[str]):
        self._weight_features = value

    @property
    def synthesize_external_bootstrap_data(self) -> bool:
        """_synthesize_external_bootstrap_data"""
        return self._synthesize_external_bootstrap_data

    @synthesize_external_bootstrap_data.setter
    def synthesize_external_bootstrap_data(self, value: bool):
        self._synthesize_external_bootstrap_data = value
