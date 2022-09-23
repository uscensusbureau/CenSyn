import copy
import logging
import math
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .encoder import Encoder
from censyn.utils import remove_indicated_values


class NumericalEncoder(Encoder):
    """
    Numerical Encoder also known as Label Encoder converts each text value to numeric. It is useful
    for both nominal (values without order) and ordinal (values with order or rank) categorical data.
    """

    def __init__(self, *args, mapping: Dict = None, alpha: float = None, **kwargs) -> None:
        """
        Initializer of the NumericalEncoder.

        :param: args: Positional arguments passed on to super.
        :param: mapping: Dictionary mapping of the categorical to numerical encoding.
        :param:alpha: Float to compute the number of encoded columns. It must be greater than or equal to 0.
        :param: kwargs: Keyword arguments passed on to super.
        :raise: ValueError: When alpha is less than 0.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate NumericalEncoder for {self.column}. mapping {mapping} and alpha {alpha}.")

        self._mapping = mapping if mapping else {}
        if alpha is not None and alpha < 0:
            raise ValueError('alpha is less than 0.0')
        self._alpha = alpha
        self._number_encode_columns = 1
        self._encode_names = super().encode_names
        self._encode_dict = None
        self._decode_mapping = None
        self._set_encode_columns()

    @property
    def encode_names(self) -> List[str]:
        """List of the encoded column names"""
        return self._encode_names

    @property
    def mapping(self) -> Dict:
        """The dictionary mapping of the categorical to numerical encoding."""
        return self._mapping

    @property
    def alpha(self) -> float:
        """
        Alpha is used to compute the number of encoded columns for a categorical variable by
        multiplying the total number of possible values for the variable by a constant alpha
        which is taken to be between 0 and 1.0.
        """
        if self._alpha is None:
            logging.debug(msg=f"NumericalEncoder.alpha() for {self.column} calculate value.")
            bins = [0, 6, 11, 18, 30, 45, 70, 105, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            len_mapping = len(self.mapping)
            if len_mapping == 0:
                return 0.0
            try:
                encode_size = np.digitize([2.0 * len_mapping], bins)[0]
            except IndexError:
                encode_size = len(bins)
            self._alpha = encode_size / len_mapping
        elif math.isinf(self._alpha):
            return 0.0
        return self._alpha

    def _set_encode_columns(self) -> None:
        """Set the number of encode columns and the encode_names."""
        logging.debug(msg=f"NumericalEncoder._set_encode_columns() for {self.column}.")
        self._number_encode_columns = max(int(round(self.alpha * len(self.mapping))), 1)

        # Encoded column names are the 'column' with '_1', '_2', ... or '_n' appended when more than 1.
        if self._number_encode_columns == 1:
            self._encode_names = super().encode_names
        else:
            self._encode_names = [f'{self.column}_{i}' for i in range(self._number_encode_columns)]

    def encode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Method to encode and create indicator data.

        :param: in_df: The input DataFrame data to encode.
        :return: The resulting encoded data.
        """
        logging.debug(msg=f"NumericalEncoder.encode() for {self.column}. in_df {in_df.shape}.")
        self.validate_encode(in_df)

        encode_df = in_df if self.inplace else pd.DataFrame(index=in_df.index)

        if self.indicator:
            if self.inplace:
                self.create_indicator(encode_df)
            else:
                encode_df = pd.concat([encode_df, self.create_indicator(in_df)], axis=1, join='outer')

        if not self.mapping:
            values = in_df[self.column].unique()
            values = np.sort([x for x in values if x is not None])
            self._mapping = {values[i]: i for i in range(len(values))}
            self._set_encode_columns()

        # Update mapping for proper data type.
        if np.issubdtype(in_df[self.column].dtype, np.number):
            warned = False
            for k, v in self._mapping.items():
                if not np.issubdtype(type(k), np.number):
                    if not warned:
                        logging.warning(f"Feature {self.column} encoding numeric data with non numeric keys.")
                        warned = True
                    self._mapping[int(k)] = v
                    del self._mapping[k]
        if self._number_encode_columns == 1:
            # if just a single column replace is best.
            if not in_df[self.column].isnull().values.all():
                encode_df[self.column] = in_df[self.column].replace(self._mapping)
            else:
                encode_df[self.column] = in_df[self.column]
        else:
            # Multiple encode columns.
            # construct list of rows of _number_encode_columns random permutations of mapping values,
            # each to correspond to a different possible ordering of the values of the 'column' variable.
            if not self._encode_dict:
                encoding_matrix = []
                for k in range(self._number_encode_columns):
                    row = list(self.mapping.values())
                    np.random.shuffle(row)
                    encoding_matrix += [row]

                # Take transpose so that each row becomes a column of length _number_encode_columns:
                encoding_matrix = np.array(encoding_matrix).T
                # Put the rows (giving the encoding for each possible value) into a dictionary:
                self._encode_dict = dict(zip(self.mapping.keys(), encoding_matrix))

            # Set all new encoding column values to np.nan
            index = pd.Index(in_df.columns).get_loc(self.column) if self.inplace else len(encode_df.columns)
            for c_names in self.encode_names:
                encode_df.insert(loc=index, column=c_names, value=np.nan)
                index = index + 1

            for x in self.mapping.keys():
                # mask is True when original variable value equals x:
                mask = (in_df[self.column] == x)
                # Where mask is True sets new encoded columns to be corresponding
                # array in encoding dictionary for variable:
                if np.any(mask):
                    encode_df.loc[mask, self.encode_names] = self._encode_dict[x]

            # If inplace remove the original data.
            if self.inplace:
                encode_df.drop(labels=self.column, axis=1, inplace=True)

        for column_name in self.encode_names:
            encode_df[column_name].fillna(value=0, inplace=True)
        if self.inplace:
            return None
        return encode_df

    def decode(self, in_df: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """
        Decodes the data. This is the inverse of the encode method.

        :param: in_df: The input DataFrame data to decode.
        :return: The decoded data.
        """
        logging.debug(msg=f"NumericalEncoder.decode() for {self.column}. in_df {in_df.shape}.")
        # Create the mapping for decoding.
        if self._decode_mapping is None:
            self._decode_mapping = {v: k for k, v in self.mapping.items()}

        decode_df = in_df if self.inplace else pd.DataFrame(index=in_df.index)

        if self._number_encode_columns == 1:
            # if just a single column replace is best.
            decode_df[self.column] = in_df[self.column].replace(self._decode_mapping)
        else:
            # Multiple encoded columns.
            if self.inplace:
                index = pd.Index(in_df.columns).get_loc(self.encode_names[0])
                decode_df.insert(loc=index, column=self.column, value=np.nan)
            else:
                decode_df[self.column] = pd.Series(np.nan, index=in_df.index)
            for v in self._encode_dict.keys():
                enc_cols = self.encode_names
                mask = pd.Series(True, index=in_df.index, dtype='bool')
                for c_, v_ in zip(enc_cols, self._encode_dict[v]):
                    mask = mask & (in_df.loc[:, c_] == v_)
                if any(mask):
                    decode_df.loc[mask, self.column] = v
            if self.inplace:
                decode_df.drop(labels=self.encode_names, axis=1, inplace=True)

        if self.indicator:
            remove_indicated_values(self.indicator_name, self.column, in_df, decode_df)
        if self.inplace:
            self.drop_indicator(decode_df)
            return None
        return decode_df

    def to_dict(self) -> Dict:
        """The NumericalEncoder's attributes in dictionary for use in processor configuration."""
        attr = {'mapping': self._mapping, 'alpha': self.alpha}
        attr.update(super().to_dict())
        return attr

    def __deepcopy__(self, memo) -> Encoder:
        logging.debug(msg=f"NumericalEncoder.__deepcopy__() for {self.column}.")
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
