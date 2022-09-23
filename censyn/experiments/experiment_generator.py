import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import product
from typing import Dict, List, Generator, Union

from censyn.encoder import Encoder
from censyn.experiments import Experiment
from censyn.features import Feature
from censyn.features import ModelChoice, ModelSpec
from censyn.models import Model, NoopModel, RandomModel, DecisionTreeModel, DecisionTreeRegressorModel
from censyn.models import HierarchicalModel, CalculateModel


class ExperimentGenerator:
    def __init__(self, feature_spec: Dict):
        """
        Initializer for the Experiment Generator.

        :param: feature_spec: This is a dict of Features that contains information about the feature and how you
        want your experiments to be generated.
        """
        logging.debug(msg=f"Instantiate ExperimentGenerator with feature_spec {feature_spec}.")
        self._feature_spec = feature_spec

    def _get_feature_spec_combos(self, feature: Feature) -> List[Feature]:
        """
        :param: feature: The feature you want to find all distinct combinations of.
        """
        feature_spec = []

        if feature.model_type is None or feature.encoder is None:
            raise ValueError('Experiment Generator requires all features have an encoder and model_type.')

        feature_spec.append(feature.model_type)
        feature_spec.append(feature.encoder)
        if feature.dependencies is not None and len(feature.dependencies) > 0:
            feature_spec.append(feature.dependencies)
        feature_spec = list(product(*feature_spec))

        return [self._feature_spec_to_feature(possible, feature) for possible in feature_spec]

    @staticmethod
    def _feature_spec_to_feature(spec_to_feature: List, feature: Feature) -> Feature:
        """
        Taking a single spec and make a feature.

        :param: spec_to_feature: this a list that has a list of dependencies, a ModelSpec, and an encoder.
        :param: feature: The original feature
        """
        dependencies = next((x for x in spec_to_feature if isinstance(x, list)), None)
        model_type = next((x for x in spec_to_feature if isinstance(x, ModelSpec)), None)
        encoder = next((x for x in spec_to_feature if isinstance(x, Encoder)), None)
        return Feature(feature_name=feature.feature_name, feature_type=feature.feature_type,
                       encoder=encoder, model_type=model_type,
                       dependencies=dependencies)

    def _model_selector(self, features: List[Feature]) -> Union[None, Dict]:
        """
        Given a list of features we return a list of their corresponding models.

        :param: features: This is the list of the features that you want to resolve to a model.
        :returns: Returns None if there is a bad model makeup such as a Noop Model that has dependencies.
        """
        models_to_return = {}
        for feature in features:
            model = self._generate_model(feature)
            if not model:
                return None
            models_to_return[feature.feature_name] = model

        return models_to_return

    @staticmethod
    def _generate_model(feature: Feature) -> Union[None, Model]:
        """
        Generate a Model from the Feature and validate the dependencies.

        :param: feature: The feature that you want to resolve to a model.
        :return: The generated model.
        """
        model = None
        try:
            current_model_spec = feature.model_type
            if current_model_spec.model is ModelChoice.NoopModel:
                model = NoopModel(target_feature=feature)
            elif current_model_spec.model is ModelChoice.RandomModel:
                model = RandomModel(target_feature=feature)
            elif current_model_spec.model is ModelChoice.DecisionTreeModel:
                sklearn_params = current_model_spec.model_params.get('sklearn', None)
                if sklearn_params is None:
                    sklearn_params = current_model_spec.model_params.copy()
                endpoints = sklearn_params.get('endpoints', None)
                if endpoints is not None:
                    del sklearn_params['endpoints']
                model = DecisionTreeModel(target_feature=feature, endpoints=endpoints,
                                          sklearn_model_params=sklearn_params)
            elif current_model_spec.model is ModelChoice.DecisionTreeRegressorModel:
                sklearn_params = current_model_spec.model_params.copy()
                endpoints = sklearn_params.get('endpoints', None)
                if endpoints is not None:
                    del sklearn_params['endpoints']
                model = DecisionTreeRegressorModel(target_feature=feature, endpoints=endpoints,
                                                   sklearn_model_params=sklearn_params)
            elif current_model_spec.model is ModelChoice.HierarchicalModel:
                model = HierarchicalModel(target_feature=feature,
                                          hierarchy_map=current_model_spec.model_params['hierarchy_map'],
                                          sklearn_model_params=current_model_spec.model_params['sklearn'])
            elif current_model_spec.model is ModelChoice.CalculateModel:
                expression = current_model_spec.model_params['expr']
                if isinstance(expression, List):
                    expression = " ".join(expression)
                model = CalculateModel(target_feature=feature, expr=expression)
            return model
        except RuntimeError:
            return None

    def _expand_features(self) -> None:
        """Expand the feature specs for all the features."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            for feature_name in self._feature_spec.keys():
                executor.submit(self._expand_feature, feature_name)

    def _expand_feature(self, feature_name: str) -> None:
        """
        :param: feature_name: Given a feature name that is in _feature_spec it does a product to find all
        the distinct possibilities of that feature.
        """
        current_feature = self._feature_spec[feature_name]
        self._feature_spec[feature_name] = self._get_feature_spec_combos(current_feature)

    @staticmethod
    def _product_dict(dicts: Dict) -> Generator:
        """
        This function that takes a dict and does a product on all the values.

        :param: dicts: Does a product on the values of this dictionary.
        """
        return (dict(zip(dicts, deepcopy(x))) for x in product(*dicts.values()))

    def generate(self) -> List[Experiment]:
        """
        This is the main function of the class. It takes the feature_spec ie Dict of features feature_name to feature.
        It takes each feature and expands it to all possible combinations of that feature using a cartesian product of
        model_type, encoder, and dependencies. So if a feature has dependencies [[f1,f2], None] encoder = [e1, e2] this
        will create 4 possible features. During the model selection process we do some validation to make sure that all
        the models are valid. Ie a random model should not have dependencies.
        """
        self._expand_features()

        experiments_to_return = []
        for idx, experiment in enumerate(self._all_experiments()):
            models = self._model_selector(experiment.values())
            if models is not None:
                experiments_to_return.append(Experiment(experiment_name=f'experiment-{idx}',
                                                        experiment_models=models))
        return experiments_to_return

    def _all_experiments(self) -> List:
        """
        Create a list of all the experiments from the feature spec.

        :return: List of experiments.
        """
        if not all(isinstance(value, Feature) for value in self._feature_spec.values()):
            all_experiments = list(self._product_dict(self._feature_spec))
        else:
            all_experiments = [self._feature_spec]
        return all_experiments
