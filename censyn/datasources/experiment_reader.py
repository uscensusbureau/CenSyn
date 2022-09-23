import dill
import json
import logging
from typing import Dict, List

from .datasource import FileSource
from censyn.experiments import ExperimentGenerator, Experiment
from censyn.utils import find_class, resolve_abs_file


class ExperimentFile(FileSource):
    def __init__(self, path_to_file: str):
        """
        Initialize the Experiment File.

        :param: path_to_file: String the file path.
        """
        super().__init__(path_to_file)
        logging.debug(msg=f"Instantiate ExperimentFile with path_to_file {path_to_file}.")

        self._file_content = json.loads(self.file_handle.read())

        if not self._valid_file():
            msg = f"The sections features and report are required in the experiment file reader."
            logging.error(msg=msg)
            raise ValueError(msg)

    def _valid_file(self) -> bool:
        """
        Checks to see if the file has the required fields.
        
        :return: Boolean if it is a valid file or not.
        """
        logging.debug(msg=f"ExperimentFile._valid_file().")
        for experiment in self._file_content.get('experiments'):
            if 'features' not in experiment or 'report' not in experiment:
                return False
        return True

    @staticmethod
    def _resolve_target_features(un_pickled: Dict, features: Dict):
        """After we read the pickled file we need to resolve the target_features for the models."""
        logging.debug(msg=f"ExperimentFile._resolve_target_features(). un_pickled {un_pickled} and "
                          f"features {features}.")
        for feature_name, model in un_pickled.items():
            model.target_feature = features[feature_name]
            if model.indicator_model:
                model.indicator_model.target_feature = features[feature_name]

    def to_experiments(self) -> List[Experiment]:
        """
        This generates a list of experiments.
        
        :return: List[Experiment] -- This list of experiments that where read from the file.
        """
        logging.debug(msg=f"ExperimentFile.to_experiments().")
        experiment_list = []
        for experiment in self._file_content.get('experiments', []):
            experiment_list.extend(self._to_experiment(experiment))
        return experiment_list

    def _to_experiment(self, experiment: Dict) -> List[Experiment]:
        """
        Create experiments from an experiment dictionary. This is done be reading in a pickled experiment of by
        generating experiments from the feature specification.

        :param: experiment: An dictionary of an experiment.
        :return: The list of experiments.
        """
        logging.debug(msg=f"ExperimentFile._to_experiment(). experiment {experiment}.")
        feature_dict = self._feature_dict(experiment)

        pickled_model = experiment.get('pickled_model')
        if pickled_model:
            return [self._from_pickled_model(pickled_model, feature_dict)]

        experiment_generator = ExperimentGenerator(feature_spec=feature_dict)
        return experiment_generator.generate()

    def _feature_dict(self, experiment: Dict) -> Dict:
        """
        Create the feature dictionary.

        :param: experiment: The experiment dictionary to read.
        :return: the feature dictionary.
        """
        logging.debug(msg=f"ExperimentFile._feature_dict(). experiment {experiment}.")
        features = experiment.get('features')
        if features:
            feature_file_name = resolve_abs_file(path_to_file=self.path_to_file,
                                                 file_name=features.get('file_name'))
            features = find_class(FileSource, features.get('file_source'),
                                  {'path_to_file': feature_file_name}).feature_definitions
            return {feature.feature_name: feature for feature in features}
        return {}

    def _from_pickled_model(self, pickled_model: Dict, feature_dict: Dict) -> Experiment:
        """
        Generate an experiment from the pickled model.

        :param: pickled_model: the pickled model dictionary.
        :param: feature_dict: the feature dictionary.
        :return: The created Experiment.
        """
        logging.debug(msg=f"ExperimentFile._from_pickled_model(). pickled_model {pickled_model} and "
                          f"feature_dict {feature_dict}.")
        model_file_name = resolve_abs_file(path_to_file=self.path_to_file,
                                           file_name=pickled_model.get('file_name'))
        experiment_name = pickled_model.get('experiment_name')
        with open(model_file_name, 'rb') as dill_file:
            un_pickled_models = dill.load(dill_file)
            self._resolve_target_features(un_pickled_models, feature_dict)
            return Experiment(experiment_name=experiment_name, experiment_models=un_pickled_models)
