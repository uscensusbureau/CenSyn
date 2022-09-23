import datetime
import json
import logging
from typing import Dict, List

from .censyn_base import CenSynBase
from censyn.datasources import save_data_source, validate_data_source
from censyn.results import ResultLevel, MappingResult
from ..utils.config_util import config_string, config_list, config_filters

HEADER_TEXT: str = 'Convert Summary: \n' \
                   'The convert was run on {start_time} and took {duration} seconds. \n'

# Configuration labels
FILTERS: str = 'filters'
DATA_FILES: str = 'data_files'
OUTPUT_FILE: str = 'output_file'
ALL_CONFIG_ITEMS = [FILTERS, DATA_FILES, OUTPUT_FILE]


class Convert(CenSynBase):
    def __init__(self, config_file: str) -> None:
        """Initialize the CenSyn Convert class."""
        super().__init__()
        logging.debug(msg=f"Instantiate Convert with config_file {config_file}.")

        self._data_files = []
        self._filters = []
        self._output_file = ''

        if config_file:
            with open(config_file, 'r') as w:
                config = json.loads(w.read())
                self.load_config(config)

        # Create report
        self.initialize_report()

        # Validate the settings
        if self._output_file:
            if not validate_data_source(self._output_file):
                msg = f"Must provide a supported output data file. Found {self._output_file} instead."
                logging.error(msg=msg)
                raise ValueError(msg)

    @property
    def all_config_items(self) -> List[str]:
        """Getter for the list of configuration items."""
        config_items = ALL_CONFIG_ITEMS.copy()
        config_items.append(super().all_config_items)
        return config_items

    def execute(self) -> None:
        """Execute the CenSyn convert process."""
        logging.debug(msg=f"Convert.execute().")
        e_start_time = datetime.datetime.now()

        self.load_features()

        if self.features:
            # validate dependencies for the features
            self.validate_features(self.features)
            self.validate_filter_features(filters=self._filters, features=self.features)

        # Load the data
        logging.info('Loading input files...')
        input_df = self.report.time_function('Load-data-time', self.load_data, {'file_names': self._data_files,
                                                                                'features': self.features})
        logging.info('Generate the input data.')
        input_df = self.report.time_function('Generate-data-time', self.generate_data, {'data_df': input_df,
                                                                                        'features': self.features})
        # Filter the data
        logging.info('Filter the input data.')
        input_df = self.report.time_function('Filter-data-time', self.filter_data, {'filters': self._filters,
                                                                                    'data_df': input_df})

        logging.info(f'Writing to output file {self._output_file}.')
        save_data_source(file_name=self._output_file, in_df=input_df)
        logging.info(f'Finished writing to output file {self._output_file}.')

        # Configuration result
        config_result = MappingResult(value=self.get_config(), level=ResultLevel.GENERAL,
                                      metric_name="Convert configuration")
        self.report.add_result("Convert configuration", config_result)

        # Create the output report
        logging.info('Create report')
        e_duration = datetime.datetime.now() - e_start_time
        self.report.header = HEADER_TEXT.format(start_time=e_start_time,
                                                duration=e_duration.total_seconds())
        self.report.produce_report()
        logging.info('Finish convert')

    def get_config(self) -> Dict:
        """
        Get the configuration dictionary.

        :return: configuration dictionary:
        """
        logging.debug(msg=f"Convert.get_config().")
        filters_config = [{'class': f.__class__.__name__, 'attributes': f.to_dict()} for f in self._filters]
        config = {
            FILTERS: filters_config,
            DATA_FILES: self._data_files,
            OUTPUT_FILE: self._output_file,
        }
        config.update(super().get_config())
        return config

    def load_config(self, config: Dict) -> None:
        """
        Load the configuration.

        :param: config: Dictionary of configurations.
        :return: None
        """
        logging.debug(msg=f"Convert.load_config(). config {config}.")
        super_config = {}
        for key, v in config.items():
            if key not in ALL_CONFIG_ITEMS:
                if key in super().all_config_items:
                    super_config[key] = v
                else:
                    msg = f"Non-supported key {key} in the configuration."
                    logging.debug(msg=msg)
                    raise ValueError(msg)
        super().load_config(config=super_config)

        # Configure the filters
        self._filters = config_filters(config=config, key=FILTERS, default_value=[])

        self._data_files = config_list(config=config, key=DATA_FILES, data_type=str, default_value=[])
        self._output_file = config_string(config=config, key=OUTPUT_FILE, default_value=None)
