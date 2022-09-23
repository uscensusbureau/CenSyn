import logging
from random import choices
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as TreeClassifier

from .models import Model
from .model_utils import feature_usage_result
from censyn.results import Result, ResultLevel, ModelResult, TableResult


def target_variable_name(target_name: str, level: int) -> str:
    """
    The hierarchical names will be in the format of 'target_name_h_n' where n is
    the level of hierarchical. Example: target_h_0, target_h_1, ...target_h_7.

    :param: target_name: The target name for this hierarchical model.
    :param: level: The hierarchical level for this tree node.
    :return: The target variable name.
    """
    return f'{target_name}_h_{str(level)}'


class HierarchicalTreeNode:
    def __init__(self,
                 var_name: str,
                 sklearn_model_params: Dict,
                 level: int = 0,
                 leaf_id: int = -1):
        """
         Each TreeNode can spawn a tree in which each node contains a model. Tree gets initialized
        (all possible nodes are created based on fitted model) when fit method is called; fit method
        also train model on the target data. Once tree is initialized, synthesis method call will
        return a non encoded synthetic data(dataframe) of actual target variable(not its hierarchy
        columns that contain refined categories).
        leaf_id -1 indicates the root doesn't have any parent. It is responsibility of the
        parent to add supply leaf id when initializing child TreeNode.

        :param: var_name: The target name for this hierarchical model.
        :param: sklearn_model_params: Dictionary of parameters for the sklearn model.
        Expected keys: max_depth, criterion, min_impurity_decrease
        :param: level: The hierarchical level for this tree node.
        :param: leaf_id:
        """
        self.var_name = var_name
        self._sklearn_model_params = sklearn_model_params
        self._sklearn_model_params['class_weight'] = 'balanced'
        self.level = level
        # Assumes that target var name is of form <name>_h_<level> if a column containing hierarchy
        # otherwise if actual column than private method _select_target_data updates
        # the target_var name in current tree node to the column name of the first column
        # in the DataFrame containing all targets(hierarchy columns + actual target column)
        self.target_var = target_variable_name(self.var_name, level)
        self.model = None
        self._features = None
        self.children: Dict[int, HierarchicalTreeNode] = dict()

        self._last_level: bool = False
        self._leaf_id = leaf_id
        self._leaf_ids_to_target_feature_values = {}  # populated in train(), used by synthesize()
        self._leaf_ids_to_target_weights = {}         # populated in train(), used by synthesize()

    def clear(self) -> None:
        """
        Clear the model.

        :return: None
        """
        self.model = None
        self._features = None
        self._last_level: bool = False
        self._leaf_ids_to_target_feature_values = {}  # populated in train(), used by synthesize()
        self._leaf_ids_to_target_weights = {}         # populated in train(), used by synthesize()

    def train(self, training_data: pd.DataFrame, all_targets_data: pd.DataFrame, weight: pd.Series):
        """
         A method that takes a DataFrame and trains a Model based on the values observed across
         the features in this DataFrame.

        :param: training_data: Encoded features to train this Model upon.
        :param: all_targets_data: DataFrame containing n-1 columns for denoting hierarchy of a variable in
        increasing granularity from left to right, with last column(nth column) as the actual variable.
        :param: weight: Weight of the samples.
        """
        # select current level's target data from all all the target data columns.
        # all_target_data is a
        target_data = self._select_target_data(all_targets_data).copy()
        self._features = training_data.columns

        self.model = TreeClassifier(**self._sklearn_model_params)

        count = 0
        while True:
            try:
                self.model.fit(training_data, target_data, sample_weight=weight)
            except ValueError as e:
                raise ValueError(e)
            except MemoryError as e:
                if count:
                    raise MemoryError(e)
                count += 1
                new_params = self.model.get_params()
                if new_params['max_depth']:
                    new_params['max_depth'] = new_params['max_depth'] - 1
                if new_params['max_leaf_nodes']:
                    new_params['max_leaf_nodes'] = int(new_params['max_leaf_nodes'] * 0.8)
                new_params['min_samples_leaf'] = new_params['min_samples_leaf'] * 2
                new_params['min_samples_split'] = new_params['min_samples_split'] * 2
                self.model.set_params(**new_params)
                continue
            break

        # m denotes leafs of the model we are creating in the current node
        # of the hierarchy tree. 'm' removes the confusion between leafs of
        # hierarchy tree or leafs of the classifier model tree.
        m_leaf_ids = self.model.apply(training_data)
        target_data['Leaf_ID'] = m_leaf_ids

        # get all unique values for target variable corresponding to a given leaf_id
        for leaf_id, group in target_data.groupby('Leaf_ID'):
            # get values (distribution) for target feature for given leaf_id
            unique_values = group[self.target_var].values.tolist()
            self._leaf_ids_to_target_feature_values[leaf_id] = unique_values
            if weight is not None:
                indexes = group[self.target_var].index
                unique_weights = weight.loc[indexes].values.tolist()
                self._leaf_ids_to_target_weights[leaf_id] = unique_weights

        # filter data and fit next.
        self._last_level = True if len(all_targets_data.columns) == 1 else False
        if not self._last_level:
            children_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = \
                self._filter_next_level_data(training_data, all_targets_data, weight)
            assert len(children_data) == len(target_data[self.target_var].unique())
            self._train_next(children_data)

    def synthesize(self, encoded_df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesizes the values for a feature at this node. The number of rows synthesized is informed
        by the shape of input_data. This method checks to see if the features specified in dependencies
        have been synthesized.

        :param: encoded_df: DataFrame containing the encoded features that this model is dependant upon.
        :return: Series of synthesized values for target feature.
        """
        encoded_df = encoded_df.copy()
        output = pd.Series(index=encoded_df.index, dtype=str)

        leaf_ids = self.model.apply(encoded_df)
        encoded_df['Leaf_ID'] = leaf_ids  # Add leaf ids
        has_weights = bool(self._leaf_ids_to_target_weights)

        # then update values for target feature, for the indices in the synthesized data corresponding to given leaf id
        for leaf_id, group in encoded_df.groupby('Leaf_ID'):

            # using the mapping of leaf ids to real target values, sample with replacement for the number of rows
            # (from synthesized data, given trained tree) with that leaf id
            sampled_values = choices(self._leaf_ids_to_target_feature_values[leaf_id],
                                     weights=self._leaf_ids_to_target_weights[leaf_id] if has_weights else None,
                                     k=group.shape[0])

            indices_curr_leaf_id = group.index.tolist()

            # update values of indices corresponding to curr leaf id
            output.update(pd.Series(sampled_values, index=indices_curr_leaf_id))
        encoded_df.pop('Leaf_ID')

        synthetic_data = pd.DataFrame(columns=[self.target_var])
        synthetic_data[self.target_var] = output

        if not self._last_level:
            # Synthesize data for all the next/child levels of the tree.
            next_synthetic_data: pd.DataFrame = self._synthesize_next(encoded_df, synthetic_data)
            # Check if indexes of the synthetic data generated from children is same as the
            # indexes of the top level synthetic data.
            assert synthetic_data.index.equals(next_synthetic_data.index)

            # append children synthetic target data to the current level's synthetic target data.
            synthetic_data = pd.concat([synthetic_data, next_synthetic_data], axis=1)

        return synthetic_data

    def _select_target_data(self, targets: pd.DataFrame) -> pd.DataFrame:
        """
        Select the target data for this level.

        :param: targets: DataFrame for the hierarchical targets.
        :return: The target DataFrame
        """
        if self.target_var in targets.columns:
            # Check any hierarchical variable available for modelling at
            # current level of the tree.
            return targets[[self.target_var]]

        # If no hierarchical variable available, then this should be last level of the tree, so check
        # if the actual variable is available for modelling. From root to second last level hierarchical
        # variables are modelled, and on last level of the tree only actual variable is modelled.
        self.target_var = self.var_name
        if self.target_var in targets.columns:
            target_data: pd.DataFrame = targets[[self.target_var]]
            self._last_level = True
        else:
            print([targets.columns])
            print(self.target_var)
            raise KeyError("Variable or its hierarchy variable not present "
                           "for modelling at level: " + str(self.level) + " of the Tree.")
        return target_data

    def _train_next(self, children_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]):
        """
        Train the next level of the hierarchical model.

        :param: children_data: Children HierarchicalTreeNodes for the next level.
        """
        for label, c_data in children_data.items():
            c_training_data, c_targets_data, c_weight = c_data
            child_node: HierarchicalTreeNode = HierarchicalTreeNode(self.var_name, self._sklearn_model_params,
                                                                    self.level + 1, label)
            child_node.train(c_training_data, c_targets_data, c_weight)
            self.children[label] = child_node

    def _filter_next_level_data(self,
                                training_data: pd.DataFrame,
                                targets: pd.DataFrame,
                                weight: pd.Series) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Filter the next level training data and target data for the current target.

        :param: training_data: DataFrame of the training data.
        :param: targets:  DataFrame of the target data.
        :param: weight: Weight of the samples.
        :return: Dictionary tuple of the training and target DataFrames for the hierarchical levels.
        """
        next_training_data: Dict[int, pd.DataFrame] = self._filter_next_training_data(training_data,
                                                                                      targets.loc[:, [self.target_var]])
        next_target_data: Dict[int, pd.DataFrame] = self._filter_next_target_data(targets)
        next_weight_data: Dict[int, pd.Series] = self._filter_next_weight_data(weight,
                                                                               targets.loc[:, [self.target_var]])

        children_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {
            label: (training_data, next_target_data[label], next_weight_data[label])
            for label, training_data in next_training_data.items()
        }

        return children_data

    def _filter_next_training_data(self,
                                   training_data: pd.DataFrame,
                                   target_data: pd.DataFrame) -> Dict:
        """
        Filter the next level training data for the current target.

        :param: training_data: DataFrame of the training data.
        :param: target_data: DataFrame of the target data.
        :return: Dictionary training DataFrames for the hierarchical levels.
        """
        next_level_data = {}
        for label, group in target_data.groupby(self.target_var):
            label_training_data: pd.DataFrame = training_data.loc[group.index]
            next_level_data[label] = label_training_data
        return next_level_data

    def _filter_next_target_data(self, targets: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Filter the next level target data for the current target.

        :param: targets: DataFrame of the target data.
        :return: Dictionary  training DataFrames for the hierarchical levels.
        """
        next_level_data: Dict[int, pd.DataFrame] = dict()
        current_level_target = targets.loc[:, [self.target_var]]
        targets = targets.drop(columns=self.target_var)
        for label, group in current_level_target.groupby(self.target_var):
            # print("(%s %d)" % (label, len(group.index)), end=' ')
            label_target_data: pd.DataFrame = targets.loc[group.index]
            next_level_data[label] = label_target_data

        return next_level_data

    def _filter_next_weight_data(self,
                                 weight: pd.Series,
                                 target_data: pd.DataFrame) -> Dict:
        """
        Filter the next level weight data for the current target.

        :param: weight: Series of the weight data.
        :param: target_data: DataFrame of the target data.
        :return: Dictionary weight Series for the hierarchical levels.
        """
        next_level_data = {}
        for label, group in target_data.groupby(self.target_var):
            label_weight_data: pd.Series = weight.loc[group.index] if weight is not None else None
            next_level_data[label] = label_weight_data

        return next_level_data

    def _synthesize_next(self,
                         encoded_df: pd.DataFrame,
                         synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize the next hierarchical level.

        :param: encoded_df: DataFrame containing the encoded features that this model is dependant upon.
        :param: synthetic_data: Previously synthesize data values for target feature.
        :return: DataFrame of synthesized values for target feature.
        """
        next_training_data: Dict[int, pd.DataFrame] = self._filter_next_training_data(encoded_df, synthetic_data)

        # Empty dataframe initialized with columns of representing all next levels of the tree(one var/col per level).
        next_synthetic_data: pd.DataFrame = pd.DataFrame()

        for label, next_training_chunk in next_training_data.items():
            # View of the Dataframe represented by a leaf id of the model of current node.
            encoded_data_chunk: pd.DataFrame = next_training_chunk
            # non_encoded_targets_chunk: pd.DataFrame = next_targets_data[label]
            # print(self.children.keys())
            # Synthesize the chuck of data (only target vars on all next levels) using children
            # corresponding to leaf id.
            synthetic_data_chunk: pd.DataFrame = self.children[label].synthesize(encoded_data_chunk)

            # Check if the indexes of the return synthetic data chuck are same as that of
            # the data chuck we send in to the children for synthesis.
            assert encoded_data_chunk.index.equals(synthetic_data_chunk.index)
            # Concatenate synthetic data from all children row-wise(since all same columns)
            next_synthetic_data = pd.concat([next_synthetic_data, synthetic_data_chunk])

        next_synthetic_data = next_synthetic_data.reindex(index=encoded_df.index)

        return next_synthetic_data

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        results = [ModelResult(value={'depth': self.model.get_depth(), 'leaves': self.model.get_n_leaves()},
                               metric_name="Model Description",
                               description=self.var_name,
                               level=ResultLevel.SUMMARY)]

        # Generate feature usage result
        usage_res = feature_usage_result(self.model, self.var_name, self._features, encode_names)
        if usage_res:
            results.append(usage_res)

        # Get the results from the children
        for child_node in self.children.values():
            child_results = child_node.get_results(encode_names=encode_names)
            for result in child_results:
                for r in results:
                    if r.metric_name == result.metric_name:
                        r.merge_result(result)
                        break

        return results


class HierarchicalModel(Model):

    def __init__(self, hierarchy_map: Dict = None, sklearn_model_params: Dict = None, *args, **kwargs):
        """
        Initialization for HierarchicalModel.  It acts as a facade to actual tree of sklearn TreeNode instances
        having links from parent to child TreeNode instances.  Provides fit and synthesize methods to use
        a tree of TreeNodes for fitting models(hierarchically as well) and synthesize target variable data

        :param: hierarchy_map: Dictionary of the hierarchical mapping.
        :param: sklearn_model_params: Dictionary of parameters for the sklearn model. Expected keys: max_depth,
        criterion, min_impurity_decrease
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs, needs_dependencies=True, encode_dependencies=True)

        if sklearn_model_params is None:
            sklearn_model_params = {'max_depth': 10, 'criterion': 'entropy', 'min_impurity_decrease': 1e-5}
        self._has_data = False
        self._is_object = False
        self._hierarchy_map = hierarchy_map if hierarchy_map else {}
        self._tree_root = HierarchicalTreeNode(self.feature_name, sklearn_model_params)

        if self.target_feature.dependencies is None:
            raise RuntimeError('Attempted initialization of a HierarchicalModel with no dependencies '
                               '(this Model should have some)')
        logging.debug(msg=f"Instantiate HierarchicalModel for {self._feature_name}.")

    def clear(self) -> None:
        """
        Clear the model.

        :return: None
        """
        logging.debug(msg=f"HierarchicalModel.clear() for {self._feature_name}.")
        super().clear()
        self._has_data = False
        self._is_object = False
        self._tree_root.clear()

    def _create_all_targets(self, target_data: pd.Series) -> pd.DataFrame:
        """
        Create all the hierarchical targets.

        :param: target_data: feature that the Model is being trained to synthesize.
        :return: DataFrame of the hierarchical targets.
        """
        logging.debug(msg=f"HierarchicalModel._create_all_targets() for {self._feature_name}. "
                          f"target {target_data.shape}")
        length = max([len(v) for v in self._hierarchy_map.values()])
        column_names = [target_variable_name(self.feature_name, level) for level in range(length)]
        all_targets = pd.DataFrame(data=None, index=target_data.index, columns=column_names)

        for x in self._hierarchy_map.keys():
            # mask is True when original variable value equals x:
            mask = (target_data == x)
            # Where mask is True sets new encoded columns to be corresponding
            # array in encoding dictionary for variable:
            all_targets.loc[mask, column_names] = self._hierarchy_map[x]

        if all_targets.isnull().any().any():
            indexes = all_targets[all_targets.isnull().T.any()].index
            # indexes = all_targets[all_targets[column_names[0]].isnull()].index
            missing_s = target_data.loc[indexes]
            missing_values = np.sort(missing_s.unique())
            msg = f"Invalid hierarchical mapping for Feature {self._feature_name} with {missing_s.shape[0]} " \
                  f"values of {missing_values}."
            logging.error(msg)
            raise ValueError(msg)
        all_targets[self._feature_name] = target_data
        return all_targets

    def train(self, predictor_df: pd.DataFrame, target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False):
        """
         A method that takes a DataFrame and trains a Model based on the values observed across the features in
         this DataFrame.  On successful completion of a call to train(), the trained property is set to True.

        :param: predictor_df: Encoded features to train this Model upon. Each encoded feature may be
        represented by more than one shuffled encoding of that feature.
        :param: target_series: Feature that the Model is being trained to synthesize.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        """
        logging.debug(msg=f"HierarchicalModel.train() for {self._feature_name}. Predictor {predictor_df.shape}, "
                          f"target {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        self.validate_dependencies(predictor_df)
        self._is_object = target_series.dtype == object

        if not indicator:
            self._train_indicator(predictor_df, target_series, weight=weight)

        # confirm target_data corresponds to _feature_name
        if not target_series.name == self._feature_name:
            raise ValueError(f'target_series passed to train() does not correspond to expected feature '
                             f'{self._feature_name}. {target_series.name} Instead')

        self._has_data = not predictor_df.empty
        if self._has_data:
            all_targets: pd.DataFrame = self._create_all_targets(target_series)
            self._tree_root.train(predictor_df, all_targets, weight=weight)

        if indicator:
            logging.debug(f'Finish Training model {self._feature_name} indicator.')
        else:
            logging.debug(f'Finish Training model {self._feature_name}.')
        self._trained = True

    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Synthesizes the values for a feature. The number of rows synthesized is informed by the shape of
        input_data. This method checks to see if the features specified in dependencies have been synthesized.
        On successful completion of a call to synthesize(), the synthesized property is set to True.

        :param: input_data: pd.DataFrame containing the features (encoded) that this model is dependant upon.
        :return: pd.Series of synthesized values (encoded) for target feature.
        """
        logging.debug(msg=f"HierarchicalModel.synthesize() for {self._feature_name}. input_data {input_data.shape}.")
        if not self._trained:
            raise ValueError(f'Model {self._feature_name} is not trained')
        self.validate_dependencies(input_data)

        output = self.output_series(data=None, indexes=input_data.index)

        if self.indicator_model:
            input_data = self._synthesize_indicator_data(input_data)
            if input_data.empty:
                logging.debug(f'synthesize model {self._feature_name} indicator empty input data.')
                return output

        if self._has_data:
            synthetic_data: pd.DataFrame = self._tree_root.synthesize(input_data)
            output = synthetic_data[self._feature_name]
        return output

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"HierarchicalModel.get_results() for {self._feature_name}. encode_names {encode_names}.")
        results = []
        if self.indicator_model:
            results.extend(self.indicator_model.get_results(encode_names))
        if not self._has_data:
            # The tree was not trained because there was no data.
            results.append(ModelResult(value={'depth': 0, 'leaves': 0},
                                       metric_name="Model Description",
                                       description=self._feature_name,
                                       level=ResultLevel.SUMMARY))
        else:
            results.extend(self._tree_root.get_results(encode_names=encode_names))
        return results
