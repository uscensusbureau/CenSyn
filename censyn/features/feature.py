import copy
import logging
from enum import Enum
from typing import List, Union
import numpy as np
import pandas as pd

from .model_spec import ModelChoice, ModelSpec
from censyn.checks import NumericDataCalculator, StringDataCalculator, ParseError
from censyn.encoder import Encoder, Binner, OTHER_INDEX


class FeatureType(Enum):
    """
    This is an enum that represents the type of feature you are working with, ie floating_point, integer, or obj.
    """
    floating_point = 0
    integer = 1
    obj = 2


class Feature:
    """
    Performs binning and handles NaN/None values for a column in a list of dataframes.
    """

    def __init__(self, feature_name: str, feature_type: FeatureType, feature_format: str = None, binner: Binner = None,
                 model_type: ModelSpec = None, encoder: Encoder = None,
                 dependencies: List[str] = None, exclude_dependencies: List[str] = None) -> None:
        """
        Note: bin_list can be specified on initialization, or populated by BinningStrategy.

        :param: feature_type: enum for column type
        :param: feature_format: data calculate grammar expression for formatting the data.
        :param: binner: Binner object used for binning feature values.
            Binning will only be performed if a Binner is provided
        :param: dependencies: The specific dependencies names for this feature.
        :param: exclude_dependencies: The exclude-dependencies names for this feature.
        :param: model_type: This is either a List[ModelSpec] or a ModelSpec
        :param: encoder: The encoder that will be used to encode this feature as.
        """
        self._feature_name: str = feature_name
        self._feature_type: FeatureType = feature_type
        self._feature_format = feature_format
        self._binner: Binner = binner
        self._dependencies = dependencies if dependencies else []
        self._exclude_dependencies = exclude_dependencies if exclude_dependencies else []
        self._train_dependencies = self._dependencies
        self._model_type = model_type
        self._encoder = encoder

        # Validate feature format
        if self._feature_format:
            if isinstance(self._feature_format, List):
                feature_format = " ".join(self._feature_format)
                self._feature_format = feature_format
            try:
                calc_dependencies = self.calculate_feature_dependencies(self._feature_format)
                for dep_f in calc_dependencies:
                    if dep_f != feature_name:
                        logging.error(f"Feature {feature_name} format {feature_format} has dependency of {dep_f} .")
                        raise ValueError(f"Feature {feature_name} format {feature_format} has dependency of {dep_f} .")
            except ParseError:
                logging.error(f"Grammar parse error on {feature_name} with format expression '{feature_format}'")
                raise ParseError

        # Data calculate expression
        self._data_calc: str = ""
        if isinstance(model_type, ModelSpec):
            if model_type.model == ModelChoice.CalculateModel:
                self._data_calc = model_type.model_params.get("expr", "")
        if self._data_calc:
            try:
                calc_dependencies = self.calculate_feature_dependencies(self._data_calc)
                if len(self.dependencies) > 0:
                    for dep_f in calc_dependencies:
                        if dep_f not in self.dependencies:
                            logging.error(f"Feature {feature_name} has calculated dependency of {dep_f } which "
                                          f"is not in the dependencies.")
                            raise ValueError(f"Feature {feature_name} has calculated dependency of {dep_f } which "
                                             f"is not in the dependencies.")
                else:
                    self._dependencies = calc_dependencies
            except ParseError:
                logging.error(f"Grammar parse error on {feature_name} with calculated "
                              f"expression '{self._data_calc}'")
                raise ParseError

    def transform(self, data_frame: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Bin a data frame, based on this Feature's Binner.

        This method returns a DataFrame binned or not (if no Binner exists for this Feature).

        See the Binner.bin method for more information on exactly how bins are applied.

        :param: data_frame: The DataFrame to transform.
        :param: inplace: Whether to modify the input DataFrame. Defaults to False.
        :return: A DataFrame with values for this feature binned as appropriate.
                 Whether it's the input DataFrame is dependent upon the inplace parameter.
        """
        new_data_frame = data_frame

        if not inplace:
            new_data_frame = copy.deepcopy(data_frame)

        if self.binner is None:
            return new_data_frame

        # raise error if self._feature_name is missing from the elements of data_frames_to_transform
        if self._feature_name not in new_data_frame.columns:
            raise RuntimeError(f'Feature {self._feature_name} is missing from at least one of the data frames'
                               f' passed into Feature.transform().')

        new_data_frame[self.feature_name] = self._binner.bin(data_frame[self.feature_name])

        # Verify OTHER bin
        other_mask = (new_data_frame[self.feature_name] == OTHER_INDEX)
        if any(other_mask):
            other_df = data_frame.loc[new_data_frame[self.feature_name].loc[other_mask].index]
            other_values = np.sort(other_df[self.feature_name].unique())
            logging.warning(f"OTHER bin value for Feature {self.feature_name} with {other_df.shape[0]} "
                            f"values of {other_values}.")

        return new_data_frame

    def __str__(self) -> str:
        # the __str__ method can include the name of the feature for additional helpful information.
        return f'Feature: {self._feature_name}\n'

    def __eq__(self, other) -> bool:
        """Override of the equals function"""
        return self.__class__ == other.__class__ and self.feature_name == self.feature_name

    def __hash__(self) -> int:
        return hash((self.feature_name, self.feature_type))

    def __deepcopy__(self, memo) -> object:
        """
        This is an implementation of the deepcopy function that is used when the copy module deepcopy function
        gets called, ie copy.deepcopy(object)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def feature_name(self) -> str:
        """Getter for feature_name"""
        return self._feature_name

    @property
    def feature_type(self) -> FeatureType:
        """Getter for feature_type"""
        return self._feature_type

    @property
    def feature_format(self) -> str:
        """Getter for feature_format"""
        return self._feature_format

    @property
    def is_data_flag(self) -> bool:
        """Getter for data Flag"""
        return True if self._data_calc else False

    @property
    def binner(self) -> Binner:
        """Getter for binner"""
        return self._binner

    @property
    def dependencies(self) -> List[str]:
        """Getter for dependencies"""
        return self._dependencies

    @property
    def train_dependencies(self) -> List[str]:
        """Getter for train_dependencies"""
        return self._train_dependencies

    @train_dependencies.setter
    def train_dependencies(self, value: List[str]):
        """Setter for train_dependencies"""
        logging.debug(f'Setting Feature {self.feature_name} train_dependencies: {value}')
        assert self.feature_name not in value, f'{self.feature_name} is in train_dependencies.'
        assert len(self._train_dependencies) == 0, f'{self.feature_name} train dependencies are being reassigned ' \
                                                   f'from {self._train_dependencies} to {value}'
        assert len(set(self._exclude_dependencies) & set(value)) == 0, f"feature {self.feature_name} setting " \
                                                                       f"dependencies to an excluded " \
                                                                       f"dependencies {value}."
        self._train_dependencies = value

    @property
    def exclude_dependencies(self) -> List[str]:
        """Getter for exclude_dependencies"""
        return self._exclude_dependencies

    @property
    def model_type(self) -> ModelSpec:
        """Getter for model_type"""
        return self._model_type

    @property
    def encoder(self) -> Encoder:
        """Getter for encoder"""
        return self._encoder

    @encoder.setter
    def encoder(self, value: Encoder):
        """Setter for encoder"""
        logging.debug(f'Setting Feature {self.feature_name} encoder: {value}')
        self._encoder = value

    def calculate_feature_dependencies(self, expr: str) -> List[str]:
        """
        Calculate the feature's dependencies in the expression (the data variables).

        :param: expr: data calculate expression.
        :return: List of dependent feature names.
        """
        if self.feature_type == FeatureType.floating_point:
            data_calculator = NumericDataCalculator(feature_name=self.feature_name, expression=expr)
        elif self.feature_type == FeatureType.integer:
            data_calculator = NumericDataCalculator(feature_name=self.feature_name, expression=expr)
        else:
            data_calculator = StringDataCalculator(feature_name=self.feature_name, expression=expr)
        try:
            dependencies = data_calculator.compute_variables()
        except ParseError:
            logging.error(f"Grammar parse error on {self.feature_name} with expression '{expr}'")
            raise ParseError

        return dependencies

    def calculate_feature_data(self, data_df: pd.DataFrame, expr: str = "") -> Union[None, pd.Series]:
        """
        Calculate the feature data from the input data and expression.

        :param data_df: Data to use to calculate the feature data.
        :param expr: data calculate expression.
        :return Pandas Series for any calculate data.
        """
        expression = expr if expr else self._data_calc
        if expression:
            if self.feature_type == FeatureType.floating_point:
                data_calculator = NumericDataCalculator(feature_name=self.feature_name, expression=expression)
                return data_calculator.execute(data_df)
            elif self.feature_type == FeatureType.integer:
                data_calculator = NumericDataCalculator(feature_name=self.feature_name, expression=expression)
                series = data_calculator.execute(data_df)
                if series.dtype == 'float':
                    mod_s = pd.Series(data=series.apply(lambda x: int(x) if x == x else None), dtype=int)
                    series = mod_s
                return series
            else:
                data_calculator = StringDataCalculator(feature_name=self.feature_name, expression=expression)
                return data_calculator.execute(data_df)
        return None

    def set_series_data_type(self, in_s: pd.Series) -> pd.Series:
        """
        Set the Pandas Series data to match the feature_type value.
        Feature type of integer are stored as float to enable NA values.

        :param: in_s: Series to validate.
        :return: The Pandas Series with valid data.
        """
        if np.issubdtype(in_s.dtype, np.number):
            if self.feature_type == FeatureType.obj:
                mod_s = pd.Series(data=in_s.apply(lambda x: str(int(x)) if x == x else None),
                                  dtype=object)
                in_s = mod_s
            elif self.feature_type == FeatureType.integer and in_s.dtype == 'float':
                mod_s = pd.Series(data=in_s.apply(lambda x: int(x) if x == x else np.nan),
                                  dtype=float)
                in_s = mod_s
        elif in_s.dtype == 'object':
            if self.feature_type == FeatureType.floating_point:
                mod_s = pd.Series(data=in_s.apply(lambda x: float(x) if x is not None else np.nan),
                                  dtype=float)
                in_s = mod_s
            elif self.feature_type == FeatureType.integer:
                mod_s = pd.Series(data=in_s.apply(lambda x: int(x) if x is not None else np.nan),
                                  dtype=float)
                in_s = mod_s
        else:
            cur_data_type = in_s.dtype
            raise ValueError(f'Unsupported data type {cur_data_type} for feature {in_s.name}.')

        return in_s
