import logging
from typing import Callable, List


def stable_features_decorator(some_func: Callable) -> Callable:
    """
    This is a decorator for marginal metric stable combinations. It removes the columns that are stable from the list,
    then passes that list to the function that generated the combinations. Before the return we create a list of tuples
    that have the stable features prepended.
    """
    def stable_wrapper(self, *args, **kwargs) -> List[tuple]:
        """
        This function is what is called when we add the @stable_features_decorator to a function. It tries to grab
        the stable features, remove them from the features list, then generate the combinations. It then adds the stable
        features to all the returned combinations. If it fails at any point this would indicate that you put the
        decorator on a function that does not work, so we will throw a warning and just call the function normally.
        """
        try:
            features = kwargs.get('features')
            if self._stable_features:
                [features.remove(stable) for stable in self._stable_features]
            kwargs['features'] = features
            to_return = some_func(self, *args, **kwargs)
            if self._stable_features:
                to_return = [tuple(self._stable_features) + _ for _ in to_return]
        except (NameError, AttributeError):
            logging.warning('Using @stable_features_decorator on none marginal metric function ignoring')
            to_return = some_func(self, *args, **kwargs)
        return to_return
    return stable_wrapper
