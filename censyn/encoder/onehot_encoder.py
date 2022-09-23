import copy
import logging
from typing import Dict, List, Union

import pandas as pd

from .encoder import Encoder


class OneHotEncoder(Encoder):
    """OneHot Encoder"""

    def __init__(self, *args, mapping: Dict, **kwargs) -> None:
        """
        Initializer of the OneHotEncoder.

        :param: args: Positional arguments passed on to super.
        :param: mapping: Dictionary mapping of the categorical value to column name.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate OneHotEncoder for {self.column}. mapping {mapping}.")
        self._mapping = mapping
        self._decode_mapping = {v: k for k, v in self.mapping.items()}
        self._encode_names = list(mapping.values())

    @property
    def encode_names(self) -> List[str]:
        """List of the encoded column names"""
        return self._encode_names

    @property
    def mapping(self) -> Dict:
        """The dictionary mapping of the categorical to OneHot encoding."""
        return self._mapping

    def encode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Method to encode and create indicator data.

        :param: in_df: The input DataFrame data to encode.
        :return: The resulting encoded data.
        """
        logging.debug(msg=f"OneHotEncoder.encode() for {self.column}. in_df {in_df.shape}.")
        self.validate_encode(in_df)

        encode_df = pd.get_dummies(
            in_df[self.column].astype(pd.api.types.CategoricalDtype(categories=self.mapping.keys())),
            dummy_na=self.indicator)
        encode_df.rename(columns=self.mapping, inplace=True)
        if self.indicator:
            # Set the indicator column name which is the last column
            encode_df.columns.values[-1] = self.indicator_name
            # Reorder the columns so that the indicator is first.
            columns = [self.indicator_name]
            columns.extend(self.mapping.values())
            encode_df = encode_df[columns]

        if self.inplace:
            index = pd.Index(in_df.columns).get_loc(self.column)
            in_df.drop(columns=self.column, inplace=True)
            for i in range(len(encode_df.columns)):
                name = encode_df.columns.values[i]
                in_df.insert(loc=index + i, column=name, value=encode_df[name])
            return None

        return encode_df

    def _decode_row(self, row) -> str:
        """
        Function for the decoding of a row of the data.

        :param: row: Row of DataFrame to decode.
        :return: The string categorical value that was decoded.
        """
        for c in self.mapping.values():
            if row[c]:
                return self._decode_mapping[c]

    def decode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Decodes the data. This is the inverse of the encode method.

        :param: in_df: The input DataFrame data to decode.
        :return: The decoded data.
        """
        logging.debug(msg=f"OneHotEncoder.decode() for {self.column}. in_df {in_df.shape}.")
        decode_df = in_df if self.inplace else pd.DataFrame(index=in_df.index)
        # Create the decoded data.
        decode_series = in_df.apply(self._decode_row, axis='columns')

        if self.inplace:
            # insert decode data in the proper location.
            index = pd.Index(in_df.columns).get_loc(self.encode_names[0])
            decode_df.insert(loc=index, column=self.column, value=decode_series)
            # Delete encoded data from the input.
            decode_df.drop(labels=self.encode_names, axis=1, inplace=True)
            self.drop_indicator(decode_df)
            return None

        decode_df[self.column] = decode_series
        return decode_df

    def to_dict(self) -> Dict:
        """The OneHotEncoder's attributes in dictionary for use in processor configuration."""
        attr = {'mapping': self._mapping}
        attr.update(super().to_dict())
        return attr

    def __deepcopy__(self, memo) -> Encoder:
        logging.debug(msg=f"OneHotEncoder.__deepcopy__() for {self.column}.")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
