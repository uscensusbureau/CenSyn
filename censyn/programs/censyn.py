import argparse
import logging

import multiprocessing as mp
from typing import Dict, Tuple

from censyn.results import Result
from censyn.evaluator.evaluate import Evaluate
from censyn.programs.convert import Convert
from censyn.programs.join import Join
from censyn.synthesis.synthesize import Synthesize
from censyn.version.version import CENSYN_VERSION
from ..utils.logger_util import LOG_FORMAT, get_logging_level


class Censyn:
    """Handles the endpoints of the package."""
    def __init__(self, *args, **kwargs) -> None:
        start = mp.get_start_method(allow_none=True)
        if not start:
            mp.set_start_method('spawn')

        args, parser = self.parse_arguments(*args, **kwargs)

        logging.basicConfig(format=LOG_FORMAT, level=get_logging_level())

        self._process = None
        if args.convert_config_file is not None:
            self._process = self.create_convert(config_file=args.convert_config_file)
        elif args.eval_config_file is not None:
            self._process = self.create_evaluate(config_file=args.eval_config_file)
        elif args.join_config_file is not None:
            self._process = self.create_join(config_file=args.join_config_file)
        elif args.synthesize_config_file is not None:
            self._process = self.create_synthesize(config_file=args.synthesize_config_file)
        else:
            parser.print_help()

    @property
    def valid_process(self) -> bool:
        return self._process is not None

    def execute(self) -> None:
        """Execute process, runs one of Convert, Evaluate, Join, or Synthesize."""
        if not self._process:
            raise RuntimeError("No process defined for Censyn.")
        self._process.execute()

    def results(self) -> Dict[str, Result]:
        """Results of the process."""
        if not self._process:
            raise RuntimeError("No process defined for Censyn.")
        return self._process.report.results

    def parse_arguments(self, *args, **_) -> Tuple:
        """Creates argument parser and parses arguments, returns both."""
        parser = argparse.ArgumentParser(epilog=self.parser_epilog)
        parser.add_argument('--convert_config_file', '-c', dest='convert_config_file', type=str,
                            help='File name of convert configuration file. Defaults to convert.cfg.')
        parser.add_argument('--eval_config_file', '-e', dest='eval_config_file', type=str,
                            help='File name of evaluate configuration file. Defaults to eval.cfg.')
        parser.add_argument('--join_config_file', '-j', dest='join_config_file', type=str,
                            help='File name of join configuration file. Defaults to join.cfg.')
        parser.add_argument('--synthesize_config_file', '-s', dest='synthesize_config_file', type=str,
                            help='File name of synthesize configuration file. Defaults to synthesize.cfg.')

        if args and len(args) > 0:
            if isinstance(args[0], str):
                in_a = args[0].split()
            elif isinstance(args[0], tuple):
                in_a = None
            else:
                in_a = args
            args = parser.parse_args(args=in_a)
        else:
            args = parser.parse_args()
        return args, parser

    @property
    def parser_epilog(self) -> str:
        """Parser epilog"""
        return f'Censyn Version: {CENSYN_VERSION}'

    @staticmethod
    def create_convert(config_file: str):
        """Creates Convert process.

        :param config_file: Filename of configuration file.
        :return: Convert process.
        """
        return Convert(config_file=config_file)

    @staticmethod
    def create_evaluate(config_file: str):
        """Creates Evaluate process.

        :param config_file: Filename of configuration file.
        :return: Evaluate process.
        """
        return Evaluate(config_file=config_file)

    @staticmethod
    def create_join(config_file: str):
        """Creates Join process.

        :param config_file: Filename of configuration file.
        :return: Join process.
        """
        return Join(config_file=config_file)

    @staticmethod
    def create_synthesize(config_file: str):
        """Creates Synthesize process.

        :param config_file: Filename of configuration file.
        :return: Synthesize process.
        """
        return Synthesize(config_file=config_file)
