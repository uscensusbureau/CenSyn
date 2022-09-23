from enum import Enum
from functools import total_ordering
from typing import Union


@total_ordering
class ReportLevel(Enum):
    """The level of output for the report."""
    # Intermediate levels of reporting are not currently supported.
    SUMMARY = 1
    FULL = 5

    @property
    def name(self) -> str:
        """Name of the report level."""
        return self._name_

    # The @total_ordering decorator requires at least one sorting comparison be defined.
    def __lt__(self, other) -> bool:
        """This determines if one ReportLevel is less then another."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def get_report_level(report_level: Union[str, ReportLevel]) -> ReportLevel:
    """
    Generate a ReportLevel from either a str or ReportLevel value.

    :param report_level: Either a string or ReportLevel
    :return: The ReportLevel
    :raise ValueError on an invalid report level type.
    """
    if isinstance(report_level, str):
        return ReportLevel[report_level]
    elif isinstance(report_level, ReportLevel):
        return report_level

    raise ValueError('Invalid report level type')
