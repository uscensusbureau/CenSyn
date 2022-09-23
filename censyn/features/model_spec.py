from enum import Enum
from typing import Dict


class ModelChoice(Enum):
    """This is a enum that represents the different types of modes."""
    NoopModel = 0
    RandomModel = 1
    DecisionTreeModel = 2
    DecisionTreeRegressorModel = 3
    HierarchicalModel = 4
    CalculateModel = 5


class ModelSpec:
    def __init__(self, model: ModelChoice, model_params: Dict = None):
        """
        Initialize for the model specification.

        :param model: This will be a ModelChoice
        :param model_params: This is a dictionary of the model parameters. Ie DecisionTreeModel- max_tree_depth,
        info_gain_cutoff. Default value is None.
        """
        self._model = model
        self._model_params = model_params

    @property
    def model(self) -> ModelChoice:
        """Getter for the model."""
        return self._model

    @property
    def model_params(self) -> Dict:
        """Getter for the model parameters."""
        return self._model_params
