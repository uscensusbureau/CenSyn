from typing import Dict, List

from .result import Result


class ModelResult(Result):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Model Result.

        :param args: Positional arguments passed on to super.
        :param sort_column: The name of the column for sorting of the table
        :param kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        if not isinstance(self._value, Dict):
            raise ValueError(f"ModelResult has invalid value type.")

    def merge_result(self, result) -> None:
        """
        Merge the results.

        :param result: The result to merge.
        :return: None
        """
        depth = result.value.get('depth')
        if depth is not None:
            if self._value.get('depth', 0) < depth:
                self._value['depth'] = depth
        leaves = result.value.get('leaves')
        if leaves is not None:
            if self._value.get('leaves', 0) < leaves:
                self._value['leaves'] = leaves
        trained = result.value.get('trained', False)
        if trained and self._value.get('trained', False):
            self._value['trained'] = True
        dependencies = result.value.get('dependencies', [])
        if dependencies and self._value.get('dependencies', []):
            self._value['dependencies'] = list(set(dependencies) | set(self._value.get('dependencies', [])))

    def display_value(self) -> str:
        """The display string for the result's value."""
        trained = self.value.get('trained')
        depth = self.value.get('depth')
        leaves = self.value.get('leaves')
        if depth is not None and leaves is not None:
            if depth == 0 and leaves == 0:
                output = f'Model was not trained because there was no data.'
            else:
                output = f'Maximum depth= {depth}, Number of leaves= {leaves}'
        elif trained is not None:
            if trained:
                output = f'Model was trained.'
            else:
                output = f'Model was not trained.'
        else:
            dependencies = self.value.get('dependencies', [])
            output = f"Model dependencies {dependencies}."

        return output
