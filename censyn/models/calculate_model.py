from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from .models import Model
from censyn.results import Result, ResultLevel, ModelResult


class CalculateModel(Model):
    """
    A CalculateModel requires no training. Returns the calculated data for a feature.
    """

    def __init__(self, *args, expr: str, **kwargs) -> None:
        """
        Initialization for CalculateModel

        :param: positional arguments passed on to super.
        :param: expr: Expression to calculate the model data.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs, needs_dependencies=True, encode_dependencies=False)
        self._expression = expr
        self._dependencies = None
        logging.debug(msg=f"Instantiate CalculateModel for {self._feature_name}. Expression '{self._expression}'.")

    @property
    def dependencies(self) -> List[str]:
        """Getter for dependencies"""
        if not self._dependencies:
            logging.debug(msg=f"CalculateModel.dependencies for {self._feature_name} when not defined.")
            self._dependencies = self.target_feature.calculate_feature_dependencies(expr=self._expression)
        return self._dependencies

    def calculate_dependencies(self) -> None:
        logging.debug(msg=f"CalculateModel.calculate_dependencies for {self._feature_name}.")
        if len(self.target_feature.train_dependencies) == 0:
            self.target_feature.train_dependencies = self.dependencies

    def train(self, predictor_df: pd.DataFrame, target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False):
        """
         Train for CalculateModel does nothing but set the trained property to True.

        :param: predictor_df: Encoded features to train this Model upon. Each encoded feature may be
        represented by more than one shuffled encoding of that feature.
        :param: target_series: Feature that the Model is being trained to synthesize.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        """
        logging.debug(msg=f"CalculateModel.train() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        self._trained = True

    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Returns the calculated data for a feature, given an input DataFrame.

        :param: input_data: pd.DataFrame containing the features that this model is dependant upon.
        :return: pd.DataFrame of synthesized values for target feature.
        """
        logging.debug(msg=f"CalculateModel.synthesize() for {self._feature_name}. input_data {input_data.shape}.")
        input_data[self._feature_name] = self.target_feature.calculate_feature_data(data_df=input_data,
                                                                                    expr=self._expression)
        return input_data[self._feature_name]

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"CalculateModel.get_results() for {self._feature_name}. encode_names {encode_names}.")
        results = [ModelResult(value={'dependencies': self.dependencies},
                               metric_name="Model Description",
                               description=self._feature_name,
                               level=ResultLevel.SUMMARY)]
        if self.indicator_model:
            results.extend(self.indicator_model.get_results(encode_names))
        return results
