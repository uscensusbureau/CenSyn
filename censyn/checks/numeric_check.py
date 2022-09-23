from abc import abstractmethod
from typing import List, Union

import pandas as pd

from .boolean_check import BinaryConsistencyCheck


class NumericConsistencyCheck(BinaryConsistencyCheck):
    """
    Consistency check to validate on two numeric features to each other.
    """
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the NumericConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    @abstractmethod
    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        raise NotImplementedError


class LessThanConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the LessThanConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) < self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class LessThanEqualConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the LessThanConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) <= self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class GreaterThanConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the GreaterThanConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1, op_1, feat_2, op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) > self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class GreaterThanEqualConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the GreaterThanConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1, op_1, feat_2, op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) >= self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class EqualConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the EqualConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1, op_1, feat_2, op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) == self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class NotEqualConsistencyCheck(NumericConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the NotEqualConsistencyCheck.

        :param args: Standard arguments.
        :param feat_1: Feature name of the first feature.
        :param op_1: Operations to perform on the first feature.
        :param feat_2: Feature name of the second feature.
        :param op_2: Operations to perform on the second feature.
        :param kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1, op_1, feat_2, op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) != self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]
