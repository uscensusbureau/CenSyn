from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from censyn.results import ResultLevel, TableResult
from .models import ModelFeatureUsage


def feature_usage_result(tree: Union[DecisionTreeClassifier, DecisionTreeRegressor], feature_name: str,
                         features: List[str], encode_names: Dict) -> Union[TableResult, None]:
    """
    Create a feature usage TableResult from the decision tree.

    :param: tree: Decision tree
    :param: feature_name: Name of the feature.
    :param: features: Dependent features.
    :param: encode_names: Dictionary of encoded feature names.
    :return: TableResult
    """
    if tree.tree_.node_count > 1:
        n_features = tree.tree_.n_features
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature

        features_count = np.zeros(shape=n_features, dtype=np.int64)
        node_depth = np.zeros(shape=tree.tree_.node_count, dtype=np.int64)
        is_leaves = np.zeros(shape=tree.tree_.node_count, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                features_count[feature[node_id]] = features_count[feature[node_id]] + 1
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        feature_map = [[encode_names.get(features[i], features[i]), features_count[i],
                        tree.feature_importances_[i]]
                       for i in range(n_features) if features_count[i] > 0]
        feature_df = pd.DataFrame(data=feature_map, columns=['Feature', 'Count', 'Importance'])
        feature_df = feature_df.groupby(by='Feature', as_index=False).sum()
        feature_df.name = 'Feature Usage Values'
        return TableResult(value=feature_df, factor=feature_df.shape[0],
                           sort_column='Importance', ascending=False,
                           metric_name=ModelFeatureUsage,
                           description=f'Feature {feature_name} usage',
                           level=ResultLevel.GENERAL)
    return None
