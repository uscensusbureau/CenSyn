import logging
import random
import re
from typing import List

from censyn.features import Feature, FeatureType


class FilterFeatures:
    """
    A Filter that take a List<Feature> and returns a filter list. This filter takes a dataframe and either
    returns the same dataframe or the negation of that which would be empty.
    """
    def __init__(self, negate: bool = False) -> None:
        """
        :param: negate: The result DataFrame has the negation of the filter.
        """
        logging.debug(msg=f"Instantiate FilterFeatures with negate {negate}.")
        self._negate = negate

    @property
    def negate(self) -> bool:
        """Boolean flag to produce the inverse of the feature filter."""
        return self._negate

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List. In this case it will either hand back the same dataframe or empty dataframe.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterFeatures.execute(). features {features}.")
        return features if not self.negate else []


class FilterFirstN(FilterFeatures):
    """
    This filter will either return the first N features, or all but the first N features of a List of Features.
    """
    def __init__(self, feature_count: int, *args, **kwargs) -> None:
        """
        :param: featureCount: This is the number of features that will be returns.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterFirstN with feature_count {feature_count}.")
        self._feature_count = feature_count

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterFirstN.execute(). features {features}.")
        return features[:self._feature_count] if not self.negate else features[self._feature_count:]

    @property
    def feature_count(self) -> int:
        """Returns the feature count."""
        return self._feature_count


class FilterLastN(FilterFeatures):
    """This filter will either return the last N features, or all but the last N features."""
    def __init__(self, feature_count: int, *args, **kwargs) -> None:
        """
        :param: featureCount: This is the number of features that will be returns.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterLastN with feature_count {feature_count}.")
        self._feature_count = feature_count

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterLastN.execute(). features {features}.")
        return features[-self._feature_count:] if not self.negate else features[:len(features) - self._feature_count]

    @property
    def feature_count(self) -> int:
        """Returns the feature count."""
        return self._feature_count


class FilterFeaturesBefore(FilterFeatures):
    """
    This class will provide the capability to get all the features before a given feature or
    all of the features after a given feature. If negate is provided it
    will give you all features after a given feature.
    """
    def __init__(self, feature_name: str, *args, **kwargs) -> None:
        """
        :param: feature_name: This feature name is used as the vertex for splitting the array.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterFeaturesBefore with feature_name {feature_name}.")
        self._feature_name = feature_name

    @property
    def feature_name(self) -> str:
        return self._feature_name

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterFeaturesBefore.execute(). features {features}.")
        index = next((i for i, item in enumerate(features) if item.feature_name == self._feature_name), -1)
        return features[:index] if not self.negate else features[index + 1:]


class FilterRandomFeatures(FilterFeatures):
    """A Filter that returns a List of Features based on a random selection."""
    def __init__(self, count: int = 1, percent: float = None, sort: bool = True, *args, **kwargs) -> None:
        """
        A Filter that returns a List based on a randomization. If percent is set then the
        filter will return that percentage of records in the List. Else the filter will return
        the number count of records.

        :param: count: The number count of records.
        :param: percent: The percent of records.
        :param: sort: The records are in the same sort order as the input feature if 'True' else
        the records are in random order. Default is 'True'
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterRandomFeatures with count {count}, percent {percent} and sort {sort}.")
        self._count = count
        self._percent = percent
        self._sort = sort

    @property
    def count(self) -> int:
        """The number count of records to filter."""
        return self._count

    @property
    def percent(self) -> float:
        """The percent of records to filter."""
        return self._percent

    @property
    def sort(self) -> bool:
        """The records are in the same order as the input if true."""
        return self._sort

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterRandomFeatures.execute(). features {features}.")
        number_of_features = len(features) * self.percent if self.percent is not None else self.count
        indices = random.sample(range(len(features)), int(number_of_features))
        indices = sorted(indices) if self._sort else indices
        return [features[i] for i in indices]


class FilterByFeatureType(FilterFeatures):
    """A Filter that returns a List of Features based on the feature_type."""
    def __init__(self, feature_type: FeatureType, *args, **kwargs) -> None:
        """
        A Filter that returns a DataFrame based on the feature_type.

        :param: feature_type: The FeatureType that you want to filter on.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterByFeatureType with feature_type {feature_type}.")
        self._feature_type = feature_type

    @property
    def feature_type(self) -> FeatureType:
        return self._feature_type

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterByFeatureType.execute(). features {features}.")
        if not self.negate:
            return list(filter(lambda feature: feature.feature_type == self._feature_type, features))
        return list(filter(lambda feature: feature.feature_type != self._feature_type, features))


class FilterFeatureByRegex(FilterFeatures):
    """
    A Filter that returns a List of Features based on a regex.
    """
    def __init__(self, regex: str, *args, **kwargs) -> None:
        """
        A Filter that returns a DataFrame based on the feature_type.

        :param: regex: The regex that you want to filter the list on.
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate FilterFeatureByRegex with regex {regex}.")
        self._regex = re.compile(regex)

    def execute(self, features: List[Feature]) -> List[Feature]:
        """
        Perform filter execution on List.

        :param: features: The List of Features that the filter is performed upon.
        :return: The resulting filtered List.
        """
        logging.debug(msg=f"FilterFeatureByRegex.execute(). features {features}.")
        if not self.negate:
            return list(filter(lambda feature: re.search(self._regex, feature.feature_name), features))
        return list(filter(lambda feature: re.search(self._regex, feature.feature_name) is None, features))
