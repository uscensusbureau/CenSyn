import io
import logging
import sys
import zipfile
from abc import ABC
from pathlib import Path
from typing import List, Union


class FileSource(ABC):
    """Abstract class for extracting and serving data from disk."""

    def __init__(self, path_to_file: str, file_name_in_zip: str = None) -> None:
        """
        Initializes the FileSource object to provide data file reading functionality.

        :param path_to_file: the base path to the file.
        :param file_name_in_zip: Must be provided if the file is a zip, to indicate its internal directory structure.
        """
        logging.debug(msg=f"Instantiate FileSource with path_to_file {path_to_file} and "
                          f"file_name_in_zip {file_name_in_zip}.")

        self._zipped_file = None
        self._file_handle = None

        # Sanity checks
        if not path_to_file:
            raise ValueError('No path_to_file specified for DataSource.\nCannot generate dataframe.')

        # Verify the file
        if not Path(path_to_file).is_file():
            raise FileNotFoundError(f'Invalid file path: {path_to_file}')

        if (Path(path_to_file).suffix == '.zip') == (not file_name_in_zip):
            raise ValueError('You must provide file_name_in_zip if and only if you are trying to open a zip file.')

        self._path_to_file = path_to_file
        self._file_name_in_zip = file_name_in_zip

        try:
            if Path(path_to_file).suffix == '.zip':
                self._file_handle = self._open_file_in_zip(path_to_file, file_name_in_zip)
            else:
                self._file_handle = open(path_to_file)
        except IOError as e:
            self._file_handle = None  # to let __del__ properly clean up
            raise type(e)(f'{str(e)} Unable to open definition file [ {path_to_file} ]!\n').with_traceback(sys.exc_info()[2])

    def __del__(self) -> None:
        if self.zipped_file:
            self.zipped_file.close()
        elif self.file_handle:
            self.file_handle.close()

    def _open_file_in_zip(self, zip_file_name: str, file_name_in_zip: str):
        """
        Open zip file.

        :param zip_file_name: The zipped file name.
        :param file_name_in_zip: Name of the file in the zipped file.
        :return:
        """
        logging.debug(msg=f"FileSource._open_file_in_zip(). zip_file_name {zip_file_name} and "
                          f"file_name_in_zip {file_name_in_zip}")
        self._zipped_file = zipfile.ZipFile(zip_file_name)
        return self._zipped_file.open(file_name_in_zip)

    @property
    def file_name_in_zip(self) -> str:
        return self._file_name_in_zip

    @property
    def path_to_file(self) -> str:
        return self._path_to_file

    @property
    def zipped_file(self) -> zipfile.ZipFile:
        return self._zipped_file

    @property
    def file_handle(self) -> io.TextIOWrapper:
        return self._file_handle

    @staticmethod
    def validate_extension(extensions: Union[List[str], str]) -> None:
        """
        An __init__ decorator to ensure the extension is valid. Lets the __init__ run, then checks its file attributes.

        :param extensions: the list if valid extensions
        :raise ValueError if the file extension is not allowed.
        """
        def _validate(init) -> None:
            """
            Wraps the wrapper to allow parameter passthrough.

            :param init: The __init__ function
            :return: the wrapped init
            """
            def wrapped_init(self, *args, **kwargs) -> None:
                """
                The actual wrapper on the init.

                :param self: automatically grabbed from the instance
                :param args: the arguments to the init
                :param kwargs: the keyword arguments to the init
                :return: None, because this is basically the init
                """
                init(self, *args, **kwargs)
                file_extension: str = Path(self.path_to_file).suffix
                if self.zipped_file:
                    file_extension = Path(self.file_name_in_zip).suffix
                if (isinstance(extensions, str) and file_extension != extensions) or \
                        (isinstance(extensions, list) and file_extension not in extensions):
                    raise ValueError(f'{file_extension} is an invalid file type for {type(self).__name__}.'
                                     f' Must be one of: {extensions}')
            return wrapped_init
        return _validate
