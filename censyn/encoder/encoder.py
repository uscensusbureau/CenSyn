import copy
import logging
from abc import abstractmethod
from typing import Dict, List, Union

import pandas as pd

from censyn.utils import remove_indicated_values


class Encoder:
    """The base Encoder class."""

    def __init__(self, column: str, inplace: bool = False, indicator: bool = False, indicator_name: str = '') -> None:
        """
        Base initializer for an Encoder.

        :param: column: The column header name to encode.
        :param: inplace: Boolean flag to encode the data in the input DataFrame. Default is False.
        :param: indicator: Boolean flag to create the indicator data. Default is False.
        :param: indicator_name: The name of the indicator data. Default value is the column name
        with '_indicator' appended to it.
        """
        logging.debug(msg=f"Instantiate Encoder for {column} with inplace {inplace}, indicator {indicator} and "
                          f"indicator_name {indicator_name}.")
        self._column = column
        self._inplace = inplace
        self._indicator = indicator
        self._indicator_name = indicator_name if indicator_name else f'{column}_indicator'

    @property
    def column(self) -> str:
        """The column's header name to encode."""
        return self._column

    @property
    def encode_names(self) -> List[str]:
        """List of the encoded column names"""
        return [self.column]

    @property
    def indicator_and_encode_names(self) -> List[str]:
        """List of the indicator and encode names."""
        if self.indicator:
            in_place = [self.indicator_name]
            in_place.extend(self.encode_names)
            return in_place
        return self.encode_names

    @property
    def inplace(self) -> bool:
        """Boolean flag to create encode data in the input DataFrame."""
        return self._inplace

    @inplace.setter
    def inplace(self, value: bool):
        """
        Setter for the inplace property.

        :param: value: Boolean flag to create encode data in the input DataFrame.
        """
        self._inplace = value

    @property
    def indicator(self) -> bool:
        """Boolean flag to create the indicator data."""
        return self._indicator

    @property
    def indicator_name(self) -> str:
        """The name of the indicator data."""
        return self._indicator_name

    def create_indicator(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for the creation of an indicator column data.  The created indicator column must be located (have an
        index value) just before the encoded data columns.

        :param:in_df: The input DataFrame data for checking indicator.
        :return: The resulting indicator data.
        """
        logging.debug(msg=f"Encoder.create_indicator() for {self.column}. in_df {in_df.shape}.")
        encode_df = in_df if self.inplace else pd.DataFrame(index=in_df.index)

        # Get the index of the feature column for insertion.
        index = pd.Index(in_df.columns).get_loc(self.column) if self.inplace else 0
        encode_df.insert(loc=index, column=self.indicator_name, value=in_df[self.column].isna())
        return encode_df

    def drop_indicator(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for the dropping of the indicator column data.

        :param: in_df:  The input DataFrame data for checking indicator.
        :return: The resulting data.
        """
        logging.debug(msg=f"Encoder.drop_indicator() for {self.column}. in_df {in_df.shape}.")
        # Remove the indicator_name data.
        if self.indicator:
            in_df.drop(labels=self.indicator_name, axis=1, inplace=True)
        return in_df

    def validate_encode(self, in_df: pd.DataFrame):
        """
        Validate the encoder does not have any not available values in not creating an indicator column.

        :param: in_df: The input DataFrame data to encode.
        :raise: ValueError when there are not any available values when no indicator column.
        """
        logging.debug(msg=f"Encoder.validate_encode() for {self.column}. in_df {in_df.shape}.")
        if not self.indicator and in_df[self.column].isnull().values.any():
            raise ValueError('Found no indicator column and null values in data frame that was being encoded.')

    @abstractmethod
    def encode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Method to encode and create indicator data.

        :param: in_df: The input DataFrame data to encode.
        :return: If inplace is True return None else the resulting encoded data.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Decodes the data. This is the inverse of the encode method.

        :param: in_df: The input DataFrame data to decode.
        :return: If inplace is True return None else the decoded data.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """The Encoder's attributes in dictionary for use in processor configuration."""
        return {
            'column': self._column,
            'inplace': self._inplace,
            'indicator': self._indicator,
            'indicator_name': self._indicator_name
        }


class IdentityEncoder(Encoder):
    """Identity Encoder is useful for creating indicator data."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializer of the IdentityEncoder.

        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate IdentityEncoder for {self.column}.")

    def encode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Method to encode and create indicator data.

        :param: in_df: The input DataFrame data to encode.
        :return: If inplace is True return None else the resulting encoded data.
        """
        logging.debug(msg=f"IdentityEncoder.encode() for {self.column}. in_df {in_df.shape}.")
        self.validate_encode(in_df)

        if self.inplace:
            if self.indicator:
                self.create_indicator(in_df)
            in_df[self.column].fillna(0, inplace=True)
            return None

        encode_df = pd.DataFrame(index=in_df.index)
        if self.indicator:
            encode_df = self.create_indicator(in_df)
        encode_df[self.column] = in_df[self.column]
        encode_df.fillna(value=0, inplace=True)

        return encode_df

    def decode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Decodes the data. This is the inverse of the encode method.

        :param: in_df: The input DataFrame data to decode.
        :return: If inplace is True return None else the decoded data.
        """
        logging.debug(msg=f"IdentityEncoder.decode() for {self.column}. in_df {in_df.shape}.")
        decode_df = in_df if self.inplace else pd.DataFrame(index=in_df.index)
        if not self.inplace:
            decode_df[self.column] = in_df[self.column]

        if self.indicator:
            remove_indicated_values(self.indicator_name, self.column, in_df, decode_df)
        if self.inplace:
            self.drop_indicator(in_df)
            return None

        return decode_df

    def to_dict(self) -> Dict:
        """The IdentityEncoder's attributes in dictionary for use in processor configuration."""
        return super().to_dict()

    def __deepcopy__(self, memo) -> Encoder:
        logging.debug(msg=f"IdentityEncoder.__deepcopy__() for {self.column}.")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
