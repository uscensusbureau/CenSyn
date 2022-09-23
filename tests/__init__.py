import os
from pathlib import Path

import dotenv

# Load the local .env file from the root directory
dotenv.load_dotenv()

NAS_UNAVAILABLE_MESSAGE: str = 'NAS not available. This should not occur when running locally. '\
                               'Check that you are connected to the NAS and that your .env is configured correctly. ' \
                               'See the readme.md for more information.'


def get_nas_path() -> Path:
    """
    Look for and return the NAS_PATH environment variable as a Path object.

    :return: The Path to the NAS_PATH environment variable, or None if no such environment variable exists
    """
    nas_path: str = os.getenv('NAS_PATH')
    return None if nas_path is None else Path(nas_path)


def nas_connection_available() -> bool:
    """
    Determine if a connection to the NAS is available on this machine

    :return: True if a connection exists, and False otherwise
    """
    nas_path: Path = get_nas_path()
    return nas_path is not None and nas_path.exists()


def get_nas_file(file_name: str) -> Path:
    """
    Return the absolute path on the NAS for the entered file
    Note: This currently assumes you're looking for a test data file and looks in that directory on the NAS

    :param file_name: The name of the file to be accessed
    :return: A Path object for the file to be accessed, or None if there is no connection to the NAS
    """
    assert file_name, 'No file_name input'
    nas_path: Path = get_nas_path()
    return None if nas_path is None else nas_path / 'CENSYN' / 'PUMS_Data2016' / 'Persons_Data' / file_name
