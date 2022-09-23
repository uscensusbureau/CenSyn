import logging
from typing import Dict, List, Union

from censyn.filters import Filter
from .util import find_class


def config_boolean(config: Dict, key: str, default_value: Union[bool, None] = None) -> Union[bool, None]:
    """
    Validate the configuration for a Boolean value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = config.get(key)
            if isinstance(value, bool):
                pass
            elif isinstance(value, str):
                if str in ['True', 'TRUE', 'true']:
                    value = True
                elif str in ['False', 'FALSE', 'false']:
                    value = False
                else:
                    msg = f"ValueError config key: {key} with value of {value} must be a Boolean value."
                    logging.error(msg=msg)
                    raise ValueError(msg)
            else:
                msg = f"ValueError config key: {key} with type of {type(value)} must be a Boolean value."
                logging.error(msg=msg)
                raise ValueError(msg)
            logging.debug(msg=f"Read config key: {key} with value of {value}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with value of {config.get(key)} must be a Boolean."
            logging.error(msg=msg)
            raise ex
    else:
        value = default_value
    return value


def config_int(config: Dict, key: str, default_value: Union[int, None] = None) -> Union[int, None]:
    """
    Validate the configuration for an integer value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = int(config.get(key))
            logging.debug(msg=f"Read config key: {key} with value of {value}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with value of {config.get(key)} must be an integer."
            logging.error(msg=msg)
            raise ex
    else:
        value = default_value
    return value


def config_string(config: Dict, key: str, default_value: Union[str, None] = None) -> Union[str, None]:
    """
    Validate the configuration for a string value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = str(config.get(key))
            logging.debug(msg=f"Read config key: {key} with value of {value}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with value of {config.get(key)} must be a string."
            logging.error(msg=msg)
            raise ex
    else:
        value = default_value
    return value


def config_dict(config: Dict, key: str, default_value: Union[Dict, None] = None) -> Union[Dict, None]:
    """
    Validate the configuration for a dictionary value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = config.get(key)
            if not isinstance(value, Dict):
                msg = f"ValueError config key: {key} with value of {value} must be a dictionary."
                logging.error(msg=msg)
                raise ValueError(msg)
            logging.debug(msg=f"Read config key: {key} with value of {value}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with value of {config.get(key)} must be a dictionary."
            logging.error(msg=msg)
            raise ex
    else:
        value = default_value
    return value


def config_list(config: Dict, key: str, data_type: type, default_value: Union[List, None] = None) -> Union[List, None]:
    """
    Validate the configuration for a list value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: data_type: The data types of the elements in the list.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = config.get(key)
            if not isinstance(value, List):
                msg = f"ValueError config key: {key} with value of {value} must be a list."
                logging.error(msg=msg)
                raise ValueError(msg)
            logging.debug(msg=f"Read config key: {key} with length of {len(value)}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with type of {type(config.get(key))} must be a list."
            logging.error(msg=msg)
            raise ex
        for ele in value:
            if not isinstance(ele, data_type):
                msg = f"ValueError config key: {key} values of list must be of type {data_type}."
                logging.error(msg=msg)
                raise ValueError(msg)
    else:
        value = default_value
    return value


def config_filters(config: Dict, key: str, default_value: Union[List, None] = None) -> Union[List, None]:
    """
    Validate the configuration for a list of filters value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    filters = []
    if key in config.keys():
        try:
            value = config.get(key)
            if not isinstance(value, List):
                msg = f"ValueError config key: {key} with value of {value} must be a list of filters."
                logging.error(msg=msg)
                raise ValueError(msg)
            logging.debug(msg=f"Read config key: {key} with length of {len(value)}.")

            for filter_value in value:
                filters.append(find_class(Filter, filter_value['class'], filter_value['attributes']))
        except ValueError as ex:
            msg = f"ValueError config key: {key} with type of {type(config.get(key))} must be a list of filters."
            logging.error(msg=msg)
            raise ex
    else:
        filters = default_value
    return filters


def config_logging_level(config: Dict, key: str, default_value: Union[int, str, None] = None) -> Union[int, str, None]:
    """
    Validate the configuration for a logging level value.

    :param: config: Configuration dictionary.
    :param: key: Key name of the value.
    :param: default_value: Default value is None.
    :return: Value for the key in configuration dictionary.
    """
    if key in config.keys():
        try:
            value = config.get(key)
            if isinstance(value, int):
                test_value = logging.getLevelName(value)
                if isinstance(logging.getLevelName(test_value), int):
                    value = test_value
            elif isinstance(value, str):
                if not isinstance(logging.getLevelName(value), int):
                    msg = f"ValueError config key: {key} with value of {value} must be a valid logging level."
                    logging.error(msg=msg)
                    raise ValueError(msg)
            else:
                msg = f"ValueError config key: {key} with type of {type(value)} must be a valid logging level."
                logging.error(msg=msg)
                raise ValueError(msg)
            logging.debug(msg=f"Read config key: {key} with value of {value}.")
        except ValueError as ex:
            msg = f"ValueError config key: {key} with value of {config.get(key)} must be a valid logging level."
            logging.error(msg=msg)
            raise ex
    else:
        value = default_value
    return value
