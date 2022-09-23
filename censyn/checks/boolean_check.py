import logging
from abc import abstractmethod
from typing import Dict, List, Union

import pandas as pd

operations = {'negate': lambda x: ~x,                   # Boolean single operand
              }
pd_operations = {'abs': 'abs',
                 'isnull': 'isnull',
                 'notnull': 'notnull',
                 'add': 'add',
                 'sub': 'sub',
                 'subtract': 'sub',
                 'mul': 'mul',
                 'div': 'div',
                 'pow': 'pow',
                 'round': 'round',
                 'lt': 'lt',
                 'le': 'le',
                 'gt': 'gt',
                 'ge': 'ge',
                 'eq': 'eq',
                 'equal': 'eq',
                 'ne': 'ne',
                 'between': 'between',
                 }


class BaseConsistencyCheck:
    """Base consistency check class"""
    def __init__(self, negate: bool = False):
        """
        Base ConsistencyCheck initialize

        :param: negate: The result data to negate of the consistency check.
        """
        logging.debug(msg=f"Instantiate BaseConsistencyCheck with negate {negate}.")
        self._negate = negate

    @property
    def negate(self) -> bool:
        """
        Getter for boolean negate.
        """
        return self._negate

    @abstractmethod
    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict:
        """The ConsistencyCheck's attributes in dictionary for use in processor configuration."""
        return {
            'negate': self._negate
        }


class BinaryConsistencyCheck(BaseConsistencyCheck):
    """
    Consistency check to validate on two features to each other.
    """
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the BinaryConsistencyCheck to check two features to each other.

        :param: args: Standard arguments.
        :param: feat_1: Feature name of the first feature.
        :param: op_1: Operations to perform on the first feature.
        :param: feat_2: Feature name of the second feature.
        :param: op_2: Operations to perform on the second feature.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._feature_1 = feat_1
        self._operation_1 = op_1
        self._feature_2 = feat_2
        self._operation_2 = op_2

    @abstractmethod
    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        raise NotImplementedError

    @staticmethod
    def get_argument(in_arg: str) -> Union[float, str]:
        """
        Get an argument of the function. Can convert to float if possible.

        :param:in_arg: The argument in string format.
        :return: The argument value.
        """
        try:
            if len(in_arg) > 2 and ((in_arg[0] == "'" and in_arg[-1] == "'") or
                                    (in_arg[0] == '"' and in_arg[-1] == '"')):
                return in_arg[1: -1]
            return float(in_arg)
        except ValueError:
            return in_arg

    def _do_operation(self, in_df: pd.DataFrame, in_s: pd.Series, in_op: str) -> pd.Series:
        """
        Perform the operation on the data.

        :param: in_df: The input DataFrame data.
        :param: in_s: The input Series data.
        :param: in_op: The operation.
        :return: The modified Series data.
        """
        parts = in_op.strip().split(sep=' ')
        arguments = [self.get_argument(parts[i]) for i in range(1, len(parts))]
        if len(parts[0]) == 0:
            return in_s
        # check for custom functions
        op = operations.get(parts[0])
        if op:
            return in_s.apply(func=op, args=tuple(arguments))
        # check for pandas series functions
        pd_op = pd_operations.get(parts[0])
        if pd_op:
            func = in_s.__getattr__(pd_op)
            for i in range(len(arguments)):
                if isinstance(arguments[i], str) and arguments[i] in in_df.columns:
                    arguments[i] = in_df[arguments[i]]
            out_s = func(*arguments)
            return out_s
        raise ValueError(f'operations {in_op} is unsupported format.')

    def _do_operations(self, in_df: pd.DataFrame, feat: str, in_op: Union[str, List[str]]) -> pd.Series:
        """
        Perform the operations on the data.

        :param: in_df: The input DataFrame data.
        :param: feat: Feature name of the feature.
        :param: in_op: The collection of operations.
        :return: The modified Series data.
        """
        if isinstance(in_op, str):
            return self._do_operation(in_df=in_df, in_s=in_df[feat], in_op=in_op)
        elif isinstance(in_op, List):
            cur_s = in_df[feat]
            for i in range(len(in_op)):
                cur_s = self._do_operation(in_df=in_df, in_s=cur_s, in_op=in_op[i])
            return cur_s
        raise ValueError(f'operations {in_op} is unsupported format.')

    def _data_1(self, in_df: pd.DataFrame) -> pd.Series:
        """
        Get the Series data from the input DataFrame and modified by its operations.

        :param: in_df:  The input DataFrame data to check.
        :return: The Series data.
        """
        return self._do_operations(in_df=in_df, feat=self._feature_1, in_op=self._operation_1)

    def _data_2(self, in_df: pd.DataFrame) -> pd.Series:
        """
        Get the Series data from the input DataFrame and modified by its operations.

        :param: in_df:  The input DataFrame data to check.
        :return: The Series data.
        """
        return self._do_operations(in_df=in_df, feat=self._feature_2, in_op=self._operation_2)

    def to_dict(self) -> Dict:
        """The BinaryConsistencyCheck's attributes in dictionary for use in processor configuration."""
        attr = {'feature_1': self._feature_1,
                'operation_1': self._operation_1,
                'feature_2': self._feature_2,
                'operation_2': self._operation_2
                }
        attr.update(super().to_dict())
        return attr


class BooleanConsistencyCheck(BinaryConsistencyCheck):
    """
    Consistency check to validate on two boolean features to each other.
    """
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the BooleanConsistencyCheck to check two features to each other.

        :param: args: Standard arguments.
        :param: feat_1: Feature name of the first feature.
        :param: op_1: Operations to perform on the first feature.
        :param: feat_2: Feature name of the second feature.
        :param: op_2: Operations to perform on the second feature.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    @abstractmethod
    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        raise NotImplementedError


class AndConsistencyCheck(BooleanConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the AndConsistencyCheck.

        :param: args: Standard arguments.
        :param: feat_1: Feature name of the first feature.
        :param: op_1: Operations to perform on the first feature.
        :param: feat_2: Feature name of the second feature.
        :param: op_2: Operations to perform on the second feature.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) & self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class OrConsistencyCheck(BooleanConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the OrConsistencyCheck.

        :param: args: Standard arguments.
        :param: feat_1: Feature name of the first feature.
        :param: op_1: Operations to perform on the first feature.
        :param: feat_2: Feature name of the second feature.
        :param: op_2: Operations to perform on the second feature.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) | self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class XorConsistencyCheck(BooleanConsistencyCheck):
    def __init__(self, *args, feat_1: str, op_1: Union[str, List[str]],
                 feat_2: str, op_2: Union[str, List[str]], **kwargs) -> None:
        """
        Initializer of the XorConsistencyCheck.

        :param: args: Standard arguments.
        :param: feat_1: Feature name of the first feature.
        :param: op_1: Operations to perform on the first feature.
        :param: feat_2: Feature name of the second feature.
        :param: op_2: Operations to perform on the second feature.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, feat_1=feat_1, op_1=op_1, feat_2=feat_2, op_2=op_2, **kwargs)

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = self._data_1(in_df) ^ self._data_2(in_df)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]


class AllConsistencyCheck(BaseConsistencyCheck):
    def __init__(self, *args, features: List[str], **kwargs):
        """
        Initializer of the AllConsistencyCheck. All values must have Truth.

        :param: args: Standard arguments.
        :param: features: List of Feature name to be checked.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._features = features

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = pd.Series(data=True, index=in_df.index)
        for feat in self._features:
            if in_df[feat].dtype == bool:
                mask = mask & in_df[feat]
            else:
                mask = mask & (in_df[feat] != 0)
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]

    def to_dict(self) -> Dict:
        """The AllConsistencyCheck's attributes in dictionary for use in processor configuration."""
        attr = {'features': self._features}
        attr.update(super().to_dict())
        return attr


class AnyConsistencyCheck(BaseConsistencyCheck):
    def __init__(self, *args, features: List[str], **kwargs):
        """
        Initializer of the AnyConsistencyCheck. A value must have Truth.

        :param: args: Standard arguments.
        :param: features: List of Feature name to be checked.
        :param: kwargs: Standard keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._features = features

    def execute(self, in_df: pd.DataFrame) -> List:
        """
        Method to test consistency checks.

        :param: in_df: The input DataFrame data to check.
        :return: List of the indexes which fail the consistency check.
        """
        mask = pd.Series(data=False, index=in_df.index)
        for feat in self._features:
            mask = mask | in_df[feat]
        if self.negate:
            return in_df.index[mask]
        return in_df.index[~mask]

    def to_dict(self) -> Dict:
        """The AnyConsistencyCheck's attributes in dictionary for use in processor configuration."""
        attr = {'features': self._features}
        attr.update(super().to_dict())
        return attr
