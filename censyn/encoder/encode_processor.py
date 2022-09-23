import logging
import time
from typing import Dict, List, Union

import pandas as pd

from .encoder import Encoder


class EncodeProcessor:
    """Processor for the encoding of data."""

    def __init__(self, report: bool = False, inplace: Union[bool, None] = None):
        """
        Initializer of the EncodeProcessor.

        :param: report: Boolean flag on creating a report of the encode process.
        :param: inplace: Boolean flag to create encode data in the input DataFrame. A 'True' or 'False'
        will override the inplace property of the Encoder. While a 'None' maintains the Encoder's inplace
        property during execution.
        """
        logging.debug(msg=f"Instantiate EncodeProcessor. report {report} and inplace {inplace}.")
        self._report = report
        self._inplace = inplace
        self._encoders = []
        self._timing = 0
        self._report_df = None

    @property
    def encoders(self) -> List[Encoder]:
        """The collection of encoders."""
        return self._encoders

    @property
    def report(self) -> bool:
        """Boolean flag on creating a report of the encode process."""
        return self._report

    @property
    def inplace(self) -> Union[bool, None]:
        """Boolean flag to create encode data in the input DataFrame."""
        return self._inplace

    @inplace.setter
    def inplace(self, value: Union[bool, None]):
        """
        Setter for the inplace property.

        :param: value: Boolean flag to create encode data in the input DataFrame.
        """
        self._inplace = value
        if self.inplace is not None:
            for e in self.encoders:
                e.inplace = self.inplace

    def append_encoder(self, encoder: Encoder) -> None:
        """
        Append an Encoder to the list of encoders. Will update the encoder's inplace property if
        the inplace property is not equal to 'None'.

        :param: encoder: The Encoder to append.
        :return: None.
        """
        logging.debug(msg=f"EncodeProcessor.append_encoder(). encoder {encoder}.")
        if self.inplace is not None:
            encoder.inplace = self.inplace
        self._encoders.append(encoder)

    def execute(self, in_df: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Process each of the encoders on the input DataFrame data.

        :param: in_df: The input DataFrame data to process the encoders.
        :return: The resulting encoded data. If all the Encoders have inplace 'True' then return is None since
        all the Encoders return None and the encoded DataFrame is empty.
        """
        logging.debug(msg=f"EncodeProcessor.execute(). in_df {in_df.shape}.")
        if self.report:
            self._timing = time.time()

        # Perform the encoding process
        encode_df = pd.DataFrame(index=in_df.index)
        for e in self.encoders:
            encode_df = pd.concat([encode_df, e.encode(in_df)], axis=1)

        # Create report.
        if self.report:
            self._timing = time.time() - self._timing

            report_df = in_df if encode_df.empty else encode_df
            self._report_df = report_df.describe(include='all')
            sum_columns = report_df.sum(axis=0, numeric_only=True)
            sum_columns.name = 'sum'
            self._report_df = self._report_df.append(sum_columns)
        return None if encode_df.empty else encode_df

    def get_configuration(self) -> Dict:
        """
        Get the configuration of encode processor.

        :return: The configuration of the processor as a dictionary.
        """
        logging.debug(msg=f"EncodeProcessor.get_configuration().")
        encoders_config = [{'class': e.__class__, 'attributes': e.to_dict()} for e in self.encoders]
        config = {'encoders': encoders_config, 'report': self._report, 'inplace': self.inplace}
        return config

    def set_configuration(self, config: Dict) -> None:
        """
        Load the configuration of encode processor.

        :param: config: The configuration to load.
        :return: None.
        """
        logging.debug(msg=f"EncodeProcessor.get_configuration(). config {config}.")
        for key, value in config.items():
            if key == 'encoders':
                # Configure the encoders
                for encoder_value in value:
                    # Create each encoder
                    encoder = encoder_value['class']
                    self.append_encoder((encoder(**encoder_value['attributes'])))
            elif key == 'report':
                self._report = value
            elif key == 'inplace':
                self.inplace = value

    def get_report(self) -> Dict:
        """
        Get the report of the encode process.  Reported values include timing.

        :return: The report in a dictionary format.
        """
        rep = {}
        if self.report:
            rep['timing'] = self._timing
        return rep

    def get_report_data(self) -> pd.DataFrame:
        """
        The report DataFrame. This is initially none and is only created when the report property is 'True'.

        :return: The report DataFrame.
        """
        return self._report_df
