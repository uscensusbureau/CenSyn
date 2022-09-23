import logging
import logging.handlers
from typing import Callable, Union
import sys
import traceback
from queue import Empty, Queue

LOG_FORMAT: str = '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s'
logging_level = logging.INFO
logging_file = None


def get_logging_level() -> int:
    """
    Get the logging level. Default in INFO,

    :return: The logging level
    """
    global logging_level
    return logging_level


def set_logging_level(level: Union[int, str]) -> None:
    """
    Set the logging level.

    :param: level: Level to be set.
    :return: None
    """
    global logging_level
    level = logging.getLevelName(level)
    if isinstance(level, str):
        raise ValueError(f"Invalid logging level {level}.")
    logging_level = level
    logging.getLogger().setLevel(level=level)


def get_logging_file() -> Union[str, None]:
    """
    Get the logging output file. Default is None

    :return: The logging output file
    """
    global logging_file
    return logging_file


def set_logging_file(file_name: str) -> None:
    """
    Set the logging output file.

    :param: file_name: Name of the output file.
    :return: None
    """
    global logging_file
    logging_file = file_name


def listener_configure(log_file: Union[str, None]) -> None:
    """
    Configure the logger listener.

    :param: log_file: File name for the log messages.
    :return: None
    """
    root = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(filename=log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(fh)


# This is the listener process top-level loop: wait for logging events (LogRecords)
# on the queue and handle them, quit when you get a None for a LogRecord.
def listener_process(queue: Queue, configure: Callable, log_file: Union[str, None]) -> None:
    """
    Logger listener process.

    :param: queue: Queue of logging records.
    :param: configure: Listener configure function.
    :param: log_file: File name for the log messages.
    :return: None
    """
    configure(log_file)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Empty:
            print("Empty queue", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break
        except EOFError:
            print("EOFError", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break
        except Exception:
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break
