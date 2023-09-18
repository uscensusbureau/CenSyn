from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd

import censyn.models as md
from censyn.features import Feature, FeatureType
from censyn.encoder import IdentityEncoder
from censyn.results import Result, ResultLevel, ModelResult

ModelFeatureUsage = 'Model feature usage'

# Dictionary mapping of the feature type
feature_type_d = {FeatureType.floating_point: float, FeatureType.integer: float, FeatureType.obj: str}


class Model(ABC):
    """
    Abstract class. A Model is responsible for synthesizing data for one or more Features during the synthesis process.

    Some models require training, and some don't.  Those Models that don't require training can have their synthesize()
    method called immediately.  Those Models that require training need to be passed an input (training) DataFrame to
    train on before synthesize() can be called.

    This DataFrame may include features that have been encoded.  A List of dependencies is used to determine which
    features are used for the training (and synthesis) of a given Model.

    Once trained, a Model can synthesize values for one or more target features, according to the values of input
    (i.e. dependent) features.
    """

    def __init__(self, target_feature: Feature, needs_dependencies: bool, encode_dependencies: bool,
                 is_indicator: bool = False) -> None:
        """
        Initialization for feature Model.

        :param: target_feature: Feature that contains specifications (name, encoding, etc.) for this Model's target
        feature. The dependencies for the model can be accessed via target_feature.
        :param: needs_dependencies: Boolean that specifies if synthesize need dependencies.
        :param: encode_dependencies: Boolean that specifies if synthesize need the dependencies to be encoded.
        :param: is_indicator: Boolean that specifies if model is for indicator data.
        """
        self._target_feature = target_feature
        self._trained = False
        self._needs_dependencies = needs_dependencies
        self._encode_dependencies = encode_dependencies
        self._is_indicator = is_indicator
        self._indicator_model = None
        self._feature_name = target_feature.feature_name
        logging.debug(msg=f"Instantiate Model for {self._feature_name} with needs_dependencies "
                          f"{self._needs_dependencies}, encode_dependencies {self._encode_dependencies} and "
                          f"is_indicator {self._is_indicator}.")

    @property
    def target_feature(self) -> Feature:
        """Getter for target feature"""
        return self._target_feature

    @target_feature.setter
    def target_feature(self, feature) -> None:
        """Setter for target feature"""
        self._target_feature = feature

    @property
    def feature_name(self) -> str:
        """Getter for feature name"""
        return self._feature_name

    @property
    def trained(self) -> bool:
        """Getter for trained"""
        return self._trained

    @property
    def needs_dependencies(self) -> bool:
        """Getter for needs_dependencies"""
        return self._needs_dependencies

    @property
    def encode_dependencies(self) -> bool:
        """Getter for encode_dependencies"""
        return self._encode_dependencies

    @property
    def indicator_model(self) -> Model:
        """Getter for indicator_model"""
        return self._indicator_model

    def clear(self) -> None:
        """
        Clear the model.

        :return: None
        """
        logging.debug(msg=f"Model.clear() for {self._feature_name}.")
        self._trained = False
        self._indicator_model = None

    def calculate_dependencies(self) -> None:
        pass

    def _train_indicator(self, predictor_df: pd.DataFrame, target_series: pd.Series, weight: pd.Series = None) -> None:
        """
        Trains an indicator model based on the values observed across the features in this DataFrame.
        Will drop the non-available data from the predictor DataFrame and target Series.

        :param: predictor_df: encoded features to train this Model upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the model is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :return: None
        """
        logging.debug(msg=f"Train indicator {self._feature_name}_indicator. Predictor {predictor_df.shape}, "
                          f"target {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        if self.target_feature.encoder and self.target_feature.encoder.indicator:
            indicator_name = self.target_feature.encoder.indicator_name
            encode = IdentityEncoder(column=self._feature_name, indicator=True, inplace=True)
            e_df = pd.DataFrame(target_series)
            encode.encode(e_df)
            self._indicator_model = md.DecisionTreeModel(target_feature=self.target_feature, sklearn_params = {'max_depth': 10, 'criterion': 'gini', 'min_impurity_decrease': 1e-5}, is_indicator=True)
            self._indicator_model._feature_name = f"{self._feature_name}_indicator"
            self._indicator_model.train(predictor_df=predictor_df, target_series=e_df[indicator_name],
                                        weight=weight, indicator=True)

            # Drop the non-available data
            drop_indexes = e_df.loc[(e_df[indicator_name])].index
            if not drop_indexes.empty:
                predictor_df.drop(index=drop_indexes, inplace=True)
                target_series.drop(index=drop_indexes, inplace=True)
                if weight is not None:
                    weight.drop(index=drop_indexes, inplace=True)

    def _synthesize_indicator_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize the indicator model data and filtering the input data with the valid indicator data.

        :param: input_data: pd.DataFrame containing the columns that represent the dependant features in the real data.
        Multiple columns for the same feature possible because real data may be encoded.
        :return: DataFrame of valid synthesized indicator values for target feature.
        """
        logging.debug(msg=f"Model._synthesize_indicator_data() for {self._feature_name}. input_data {input_data}.")
        mask = self.indicator_model.synthesize(input_data.copy())
        return input_data.loc[(mask == 0), :].copy()

    @abstractmethod
    def train(self, predictor_df: pd.DataFrame, target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False):
        """
         A method that takes a DataFrame and trains a Model based on the values observed across the features in
         this DataFrame.  On successful completion of a call to train(), the trained property is set to True.

        :param: predictor_df: Encoded features to train this Model upon. Each encoded feature may be
        represented by more than one shuffled encoding of that feature.
        :param: target_series: Feature that the Model is being trained to synthesize.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        """
        raise NotImplementedError('Attempted call to abstract method Model.train()')

    @abstractmethod
    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        An abstract method that takes a DataFrame of synthesized features.
        Concrete implementations of this method will synthesize the values for a feature.
        The number of rows synthesized is informed by the shape of training_data.
        On successful completion of a call to synthesize(), the synthesized property is set to True.

        :param: input_data: pd.DataFrame containing the columns that represent the dependant features in the real data.
        Multiple columns for the same feature possible because real data may be encoded.
        :return: pd.Series of synthesized values for target feature
        """
        raise NotImplementedError('Attempted call to abstract method Model.synthesize()')

    # TODO: This is commented in-order to parallelize the training part of the synthesis process.
    # TODO: As of now, commenting this method doesn't impact anything. This method is mainly used
    # TODO: to remove the target_feature reference from an instance of type Model when an Experiment is
    # TODO: written to the file-system. Further, when a saved experiment is read back into the memory, the
    # TODO: target_feature property of each model in the read experiment should be re-assigned from the
    # TODO: feature references already available in the memory. But, the part that reads back a saved
    # TODO: experiment(datasources/experiment_reader.py:ExperimentFile) seems to be broken. And since it is
    # TODO: broken, commenting this method will not impact anything at this moment.
    # TODO: Even when the ExperimentFile datasource is fixed, we should not un-comment this method, instead
    # TODO: make sure each model's target-feature is being re-assigned from the features already available in the
    # TODO: memory, and then remove this method entirely.
    # def __getstate__(self):
    #     """
    #     This overrides the default getstate. This is because dill pickle uses this to generate a pickle. We don't want
    #     the target feature to be pickled we will regenerate the target feature when we read it back in from the file.
    #     """
    #     object_dict = self.__dict__.copy()
    #     del object_dict['_target_feature']
    #     return object_dict

    def validate_dependencies(self, in_df: pd.DataFrame):
        """
        Validate the input data frame features matches the expected dependencies specified for this model.

        :param: in_df: encoded features for this Model.
        :return: None
        :raises: ValueError: the input features do not match the dependencies for the model.
        """
        pass

    def output_series(self, data: Union[List, None], indexes: pd.index) -> pd.Series:
        """
        Create the default output Series.

        :param: data: List of values on None for default data as defined by feature type.
        :param: indexes: pd.index.
        :return: default output pd.Series for the target feature.
        """
        logging.debug(msg=f"Model.output_series() for {self._feature_name}. data {data}.")
        try:
            if self._is_indicator:
                if data is None:
                    data = np.empty(len(indexes), dtype=bool)
                output = pd.Series(data=data, index=indexes, dtype=bool)
            else:
                f_type = feature_type_d.get(self.target_feature.feature_type)
                assert f_type is not None, f"Feature {self.target_feature.feature_name} has invalid feature " \
                                           f"type {self.target_feature.feature_type}."
                if np.issubdtype(f_type, np.number):
                    if data is None:
                        data = np.nan
                    output = pd.Series(data=data, index=indexes, dtype=f_type)
                else:
                    if data is None:
                        data = np.empty(len(indexes), dtype=object)
                    output = pd.Series(data=data, index=indexes, dtype=f_type)
        except ValueError as e:
            logging.error(f"Output series for feature {self.target_feature.feature_name} with feature type "
                          f"{self.target_feature.feature_type}.")
            raise e

        return output

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"Model.get_results() for {self._feature_name}. encode_names {encode_names}.")
        raise NotImplementedError('Attempted call to abstract method Model.get_results()')


class NoopModel(Model):
    """
    A NoopModel requires no training. Returns the original data for a feature.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for NoopModel

        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs, needs_dependencies=False, encode_dependencies=True)
        self._expected = None
        logging.debug(msg=f"Instantiate NoopModel for {self._feature_name}.")

    def train(self, predictor_df: Union[pd.DataFrame, None], target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False):
        """
         Train for NoopModel is the target series.  On successful completion of a call to train(), the trained
         property is set to True.

        :param: predictor_df: Encoded features to train this Model upon. Each encoded feature may be
        represented by more than one shuffled encoding of that feature.
        :param: target_series: Feature that the Model is being trained to synthesize.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        """
        logging.debug(msg=f"NoopModel.train() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        self._expected = target_series.copy()
        self._trained = True

    def synthesize(self, input_data: Union[pd.DataFrame, None]) -> pd.Series:
        """
        Returns the original data for a feature, given an input DataFrame.

        :param: input_data: pd.DataFrame containing the dependency columns that represent the target feature in the
        real data Will not be encoded.
        :return: pd.DataFrame of synthesized values for target feature.
        """
        logging.debug(msg=f"NoopModel.synthesize() for {self._feature_name}. input_data "
                          f"{input_data.shape if input_data is not None else None}.")
        return self._expected.copy()

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"NoopModel.get_results() for {self._feature_name}. encode_names {encode_names}.")
        results = [ModelResult(value={'trained': self.trained},
                               metric_name="Model Description",
                               description=self._feature_name,
                               level=ResultLevel.SUMMARY)]
        if self.indicator_model:
            results.extend(self.indicator_model.get_results(encode_names))
        return results


class RandomModel(Model):
    """
    A RandomModel requires no training - random sample with replacement from real data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialization for RandomModel

        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs, needs_dependencies=False, encode_dependencies=True)
        self._expected = None
        self._weights = None
        logging.debug(msg=f"Instantiate RandomModel for {self._feature_name}.")

    def train(self, predictor_df: Union[pd.DataFrame, None], target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False):
        """
         Train for RandomModel is the target series.  On successful completion of a call to train(), the trained
         property is set to True.

        :param: predictor_df: Encoded features to train this Model upon. Each encoded feature may be
        represented by more than one shuffled encoding of that feature.
        :param: target_series: Feature that the Model is being trained to synthesize.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        """
        logging.debug(msg=f"RandomModel.train() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        self._expected = target_series.copy()
        self._weights = weight.div(weight.sum()) if weight is not None else None
        self._trained = True

    def synthesize(self, input_data: Union[pd.DataFrame, None]) -> pd.Series:
        """
        Returns a random weighted sample (w/ replacement) for a feature, given an input DataFrame.

        :param: input_data: pd.DataFrame containing the columns that represent the target feature in the real data.
        Will not be encoded.
        :return: pd.Series of synthesized values for target feature.
        """
        logging.debug(msg=f"RandomModel.synthesize() for {self._feature_name}. input_data "
                          f"{input_data.shape if input_data is not None else None}.")
        values = [self._expected.sample(n=1, weights=self._weights).values[0] for _ in range(len(self._expected))]
        return pd.Series(data=values, index=self._expected.index)

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"RandomModel.get_results() for {self._feature_name}. encode_names {encode_names}.")
        results = [ModelResult(value={'trained': self.trained},
                               metric_name="Model Description",
                               description=self._feature_name,
                               level=ResultLevel.SUMMARY)]
        if self.indicator_model:
            results.extend(self.indicator_model.get_results(encode_names))
        return results
