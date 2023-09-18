from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from random import choices
from typing import Dict, List, Union

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as TreeClassifier, DecisionTreeRegressor as TreeRegressor

from .models import Model
from .model_utils import feature_usage_result
from censyn.results import Result, ResultLevel, ModelResult


class DecisionTreeBaseModel(Model, ABC):
    """
    A Base Model that utilizes a decision tree to synthesize features.
    """
    def __init__(self, endpoints: List = None,  sklearn_model_params: Dict = None, *args, **kwargs) -> None:
        """
        Init for DecisionTreeBaseModel

        :param: endpoints: List of values for the end points for the model.
        :param: sklearn_model_params: Dictionary of parameters for the sklearn model. Expected keys: max_depth,
        criterion, min_impurity_decrease
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs, needs_dependencies=True, encode_dependencies=True)

        params = self.validate_sklearn_params(sklearn_model_params)
        self._endpoints = [] if endpoints is None else endpoints
        self.tree = self.get_decision_tree(params)
        self._endpoints_tree = self.get_decision_tree(params) if self._endpoints else None
        self._features = None
        self._has_data = False
        self._is_object = False
        self._output_encoded = False
        self._category = None
        logging.debug(msg=f"Instantiate DecisionTreeBaseModel for {self._feature_name}.")

    def clear(self) -> None:
        """
        Clear the model.

        :return: None
        """
        logging.debug(msg=f"DecisionTreeBaseModel.clear() for {self._feature_name}.")
        super().clear()
        self._features = None
        self._has_data = False
        self._is_object = False
        self._output_encoded = False
        self._category = None
        params = self.tree.get_params(deep=True)
        self.tree = self.get_decision_tree(params)
        self._endpoints_tree = self.get_decision_tree(params) if self._endpoints else None

    def validate_sklearn_params(self, sklearn_params: Dict = None) -> Dict:
        """
        Validate the sklearn model parameters for the decision tree.

        :param: sklearn_params: Dictionary of sklearn parameters.
        :return: Valid dictionary of sklearn parameters.
        """
        return sklearn_params

    def validate_dependencies(self, in_df: pd.DataFrame):
        """
        Validate the input data frame features matches the expected dependencies specified for this model.

        :param: in_df: encoded features for this Model.
        :return: None
        :raises ValueError: the input features do not match the dependencies for the model.
        """
        logging.debug(msg=f"DecisionTreeBaseModel.validate_dependencies() for {self._feature_name}. "
                          f"in_df {in_df.shape}")
        if in_df.empty:
            msg = f"Empty data for feature model {self._feature_name}"
            logging.error(msg=msg)
            raise ValueError(msg)

        if self._features is not None:
            if len(self._features) != len(in_df.columns):
                msg = f"Model {self._feature_name} dependencies are of different sizes {len(self._features)} != " \
                      f"{len(in_df.columns)}"
                logging.error(msg=msg)
                raise ValueError(msg)
            for i in range(len(self._features)):
                if self._features[i] != in_df.columns[i]:
                    msg = f"Model {self._feature_name} dependencies are different {self._features} != {in_df.columns}"
                    logging.error(msg=msg)
                    raise ValueError(msg)
        logging.debug(msg=f"DecisionTreeBaseModel.validate_dependencies() for {self._feature_name}.")

    def _validate_target(self, target_series: pd.Series, indicator: bool = False) -> None:
        """
        Validate the target_series matches the expected target specified for this model.

        :param: target_series: target features for this Model.
        :param: indicator: Indicator flag.
        :return: None
        :raises ValueError: the input data do not match the target for the model.
        """
        logging.debug(msg=f"DecisionTreeBaseModel._validate_target() for {self._feature_name}. "
                          f"target {target_series.shape}")
        self._output_encoded = pd.api.types.is_string_dtype(target_series)

        if target_series.empty:
            msg = f"Empty data for feature model {self._feature_name}."
            logging.error(msg=msg)
            raise ValueError(msg)
        if not indicator and not target_series.name == self._feature_name:
            msg = f"Attempting to train DecisionTreeModel, target series does not correspond to the " \
                  f"specified feature. Expected {self._feature_name}. Received {target_series.name}."
            logging.error(msg=msg)
            raise ValueError(msg)
        if indicator and not target_series.name == self.target_feature.encoder.indicator_name:
            msg = f"Attempting to train DecisionTreeModel, target series does not correspond to the  specified " \
                  f"feature. Expected {self.target_feature.encoder.indicator_name}. Received {target_series.name}."
            logging.error(msg=msg)
            raise ValueError(msg)

    def validate_trained(self) -> None:
        """Validate that the Model has been trained."""
        logging.debug(msg=f"DecisionTreeBaseModel.validate_trained() for {self._feature_name}.")
        if not self._trained:
            msg = f"Model {self._feature_name} is not trained."
            logging.error(msg=msg)
            raise ValueError(msg)

    @abstractmethod
    def get_decision_tree(self,  sklearn_params: Dict):
        """
        Abstract method to generate the decision tree for the model

        :param: sklearn_params:  Dictionary of sklearn parameters.
        :return: Decision Tree
        """
        raise NotImplementedError('Attempted call to abstract method DecisionTreeBaseModel.get_decision_tree()')

    def train(self, predictor_df: pd.DataFrame, target_series: pd.Series, weight: pd.Series = None,
              indicator: bool = False) -> None:
        """
        A method that takes a DataFrame and trains a Model based on the values observed across the features in
        this DataFrame.  On successful completion of a call to train(), the trained property is set to True.

        :param: predictor_df: encoded features to train this Model upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the Model is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :param: indicator: Boolean if an indicator feature.
        :return: None
        """
        logging.debug(msg=f"DecisionTreeBaseModel.train() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        self.validate_dependencies(predictor_df)
        self._validate_target(target_series, indicator)
        self._features = predictor_df.columns
        self._is_object = target_series.dtype == object

        if not indicator:
            self._train_indicator(predictor_df, target_series, weight)

        logging.debug(f'Training {self._feature_name} data size is {predictor_df.shape[0]}.')
        self._has_data = not predictor_df.empty
        if self._has_data:
            if self._endpoints:
                end_mask = (target_series.isin(self._endpoints))
                if end_mask.any():
                    end_target = target_series.copy()
                    end_target.loc[~end_mask] = self._not_endpoint()
                    self._train_tree(self._endpoints_tree, predictor_df, end_target, weight)
                    self._post_train_endpoints(self._endpoints_tree, predictor_df, end_target, weight)
                    end_indexes = target_series.loc[end_mask].index
                    predictor_df = predictor_df.drop(index=end_indexes, inplace=False)
                    target_series = target_series.drop(index=end_indexes, inplace=False)
                    if weight is not None:
                        weight = weight.drop(index=end_indexes, inplace=False)

            if not predictor_df.empty:
                self._train_tree(self.tree, predictor_df, target_series, weight)
                self._post_train(self.tree, predictor_df, target_series, weight)
        if indicator:
            logging.debug(f'Finish Training model {self._feature_name} indicator.')
        else:
            logging.debug(f'Finish Training model {self._feature_name}.')
        self._trained = True

    def _not_endpoint(self) -> Union[float, None]:
        """
        Get value which is not in the endpoints. This is utilized in defining the target data.

        :return: A value not in the endpoints.
        """
        return None

    def _train_tree(self, tree: Union[TreeClassifier, TreeRegressor], predictor_df: pd.DataFrame,
                    target_series: pd.Series, weight: pd.Series) -> None:
        """
        A method that takes a DataFrame and trains a tree based on the values observed across the features in
        this DataFrame.

        :param: predictor_df: encoded features to train this Model upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the Model is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight for each sample.
        :return: None
        """
        logging.debug(msg=f"DecisionTreeBaseModel._train_tree() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        count = 0
        
        while True:
            try:
                logging.debug(f'Start tree fit with params {tree.get_params()}')

                target_series = self._convert_target(target_series)
                weight_a = weight.to_numpy() if weight is not None else None
                tree.fit(predictor_df.to_numpy(), target_series.to_numpy(), sample_weight=weight_a)
            except MemoryError as e:
                count += 1
                if count > 1:
                    logging.error(msg=f"DecisionTreeBaseModel._train_tree() for {self._feature_name} MemoryError.")
                    raise MemoryError(e)
                logging.info(f'MemoryError training model {self._feature_name}.')
                new_params = tree.get_params()
                if new_params['max_depth']:
                    new_params['max_depth'] = new_params['max_depth'] - 1
                if new_params['max_leaf_nodes']:
                    new_params['max_leaf_nodes'] = int(new_params['max_leaf_nodes'] * 0.5)
                new_params['min_samples_leaf'] = new_params['min_samples_leaf'] * 2
                new_params['min_samples_split'] = new_params['min_samples_split'] * 2
                tree.set_params(**new_params)
                continue
            break
        logging.debug(f'Tree {self._feature_name} depth of {tree.get_depth()} '
                      f'with number leaves of {tree.get_n_leaves()}')

    def _convert_target(self, target_series: pd.Series) -> pd.Series:
        """
        Convert target series for training.

        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :return: Convert target series for training. Real values, unencoded.
        """
        return target_series

    def _post_train_endpoints(self, tree: Union[TreeClassifier, TreeRegressor], predictor_df: pd.DataFrame,
                              target_series: pd.Series, weight: pd.Series):
        """
        A post training of the tree method for the endpoints.

        :param: predictor_df: encoded features to train this tree upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :return: None
        """
        pass

    def _post_train(self, tree: Union[TreeClassifier, TreeRegressor], predictor_df: pd.DataFrame,
                    target_series: pd.Series, weight: pd.Series):
        """
        A post training of the tree method.

        :param: predictor_df: encoded features to train this tree upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :return: None
        """
        pass

    @abstractmethod
    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        An abstract method that takes a DataFrame of synthesized features.
        Concrete implementations of this method will synthesize the values for a feature.
        The number of rows synthesized is informed by the shape of training_data.
        On successful completion of a call to synthesize(), the synthesized property is set to True.

        :param: input_data: pd.DataFrame containing the columns that represent the dependant features in the real data.
        Multiple columns for the same feature possible because real data may be encoded.
        :return: pd.Series of synthesized values for target feature
        """
        raise NotImplementedError('Attempted call to abstract method DecisionTreeBaseModel.synthesize()')

    def get_results(self, encode_names: Dict) -> List[Result]:
        """
        Get the statistical results for the model.

        :param: encode_names: The features' encoded and indicator names.
        :return: List of results.
        """
        logging.debug(msg=f"DecisionTreeBaseModel.get_results() for {self._feature_name}. "
                          f"encode_names {encode_names}.")
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
            results.append(ModelResult(value={'depth': self.tree.get_depth(), 'leaves': self.tree.get_n_leaves()},
                                       metric_name="Model Description",
                                       description=self._feature_name,
                                       level=ResultLevel.SUMMARY))

            # Generate feature usage result
            usage_res = feature_usage_result(self.tree, self._feature_name, self._features, encode_names)
            if usage_res:
                results.append(usage_res)
        return results


class DecisionTreeModel(DecisionTreeBaseModel):
    def __init__(self, *args, **kwargs) -> None:
        """
        Init for DecisionTreeModel

        :param: sklearn_model_params: Dictionary of parameters for the sklearn model. Expected keys: max_depth,
        criterion, min_impurity_decrease
        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)

        self._leaf_ids_to_target_feature_values = {}  # populated in train(), used by synthesize()
        self._leaf_ids_to_target_weights = {}         # populated in train(), used by synthesize()
        self._leaf_ids_to_endpoints_values = {}       # populated in train(), used by synthesize()
        self._leaf_ids_to_endpoints_weights = {}      # populated in train(), used by synthesize()

        if self._target_feature.dependencies is None:
            raise RuntimeError(
                'Attempted initialization of a DecisionTreeModel with no dependencies (this Model should have some)')
        logging.debug(msg=f"Instantiate DecisionTreeModel for {self._feature_name}.")

    def validate_sklearn_params(self, sklearn_params: Dict = None) -> Dict:
        """
        Validate the sklearn model parameters for the decision tree.

        :param: sklearn_params: Dictionary of sklearn parameters.
        :return: Valid dictionary of sklearn parameters.
        """
        logging.debug(msg=f"DecisionTreeModel.validate_sklearn_params() for {self._feature_name}.")
        if sklearn_params is None:
            sklearn_params = {'max_depth': 10, 'criterion': 'entropy', 'min_impurity_decrease': 1e-5}
        sklearn_params['class_weight'] = 'balanced'
        return sklearn_params

    def get_decision_tree(self,  sklearn_params: Dict) -> TreeClassifier:
        """
        Generate the decision tree for the model

        :param: sklearn_params:  Dictionary of sklearn parameters.
        :return: Decision Tree
        """
        logging.debug(msg=f"DecisionTreeModel.get_decision_tree() for {self._feature_name}.")
        return TreeClassifier(**sklearn_params)

    def clear(self) -> None:
        """
        Clear the model.

        :return: None
        """
        logging.debug(msg=f"DecisionTreeModel.clear() for {self._feature_name}.")
        super().clear()

        self._leaf_ids_to_target_feature_values = {}  # populated in train(), used by synthesize()
        self._leaf_ids_to_target_weights = {}         # populated in train(), used by synthesize()
        self._leaf_ids_to_endpoints_values = {}       # populated in train(), used by synthesize()
        self._leaf_ids_to_endpoints_weights = {}      # populated in train(), used by synthesize()

    def _convert_target(self, target_series: pd.Series) -> pd.Series:
        """
        Convert target series for training.

        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :return: Convert target series for training. Real values, unencoded.
        """
        logging.debug(msg=f"DecisionTreeModel._convert_target() for {self._feature_name}, "
                          f"target_series {target_series}.")
        if self._output_encoded:
            self._category = pd.Series(target_series, dtype='category')
            target_series = self._category.cat.codes
        if target_series.dtype == float:
            target_series = target_series.astype(dtype='str')
        return target_series

    def _post_train_endpoints(self, tree: Union[TreeClassifier, TreeRegressor], predictor_df: pd.DataFrame,
                              target_series: pd.Series, weight: pd.Series) -> None:
        """
        A post training of the tree method. Sets the leaf ids from the target data.

        :param: predictor_df: encoded features to train this tree upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :return: None
        """
        logging.debug(msg=f"DecisionTreeModel._post_train_endpoints() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        target_df = pd.DataFrame(columns=[target_series.name, 'Leaf_ID'])
        target_df[target_series.name] = target_series
        target_df['Leaf_ID'] = tree.apply(predictor_df.to_numpy())

        # get all unique values for target feature corresponding to a given leaf_id
        for leaf_id, group in target_df.groupby('Leaf_ID'):

            # get values (distribution) for target feature for given leaf_id
            unique_values = group[target_series.name].values.tolist()
            self._leaf_ids_to_endpoints_values[leaf_id] = unique_values
            if weight is not None:
                indexes = group[target_series.name].index
                unique_weights = weight.loc[indexes].values.tolist()
                self._leaf_ids_to_endpoints_weights[leaf_id] = unique_weights

    def _post_train(self, tree: Union[TreeClassifier, TreeRegressor], predictor_df: pd.DataFrame,
                    target_series: pd.Series, weight: pd.Series) -> None:
        """
        A post training of the tree method. Sets the leaf ids from the target data.

        :param: predictor_df: encoded features to train this tree upon. Each encoded feature may be represented by
        more than one shuffled encoding of that feature.
        :param: target_series: feature that the tree is being trained to synthesize. Real values, unencoded.
        :param: weight: Weight of the samples.
        :return: None
        """
        logging.debug(msg=f"DecisionTreeModel._post_train() for {self._feature_name}. "
                          f"predictor_df {'None' if predictor_df is None else predictor_df.shape}, "
                          f"target_series {target_series.shape}, weight {'None' if weight is None else weight.shape}.")
        target_df = pd.DataFrame(columns=[target_series.name, 'Leaf_ID'])
        target_df[target_series.name] = target_series
        target_df['Leaf_ID'] = tree.apply(predictor_df.to_numpy())

        # get all unique values for target feature corresponding to a given leaf_id
        for leaf_id, group in target_df.groupby('Leaf_ID'):

            # get values (distribution) for target feature for given leaf_id
            unique_values = group[target_series.name].values.tolist()
            self._leaf_ids_to_target_feature_values[leaf_id] = unique_values
            if weight is not None:
                indexes = group[target_series.name].index
                unique_weights = weight.loc[indexes].values.tolist()
                self._leaf_ids_to_target_weights[leaf_id] = unique_weights

    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Synthesizes the values for a feature. The number of rows synthesized is informed by the shape of input_data.
        This method checks to see if the features specified in dependencies have been synthesized.
        On successful completion of a call to synthesize(), the synthesized property is set to True.

        :param: input_data: pd.DataFrame containing the features (encoded) that this model is dependant upon.
        :return: pd.Series of synthesized values (unencoded) for target feature.
        """
        logging.debug(msg=f"DecisionTreeModel.synthesize() for {self._feature_name}. input_data {input_data.shape}.")
        self.validate_trained()
        self.validate_dependencies(input_data)

        output = self.output_series(data=None, indexes=input_data.index)

        # Synthesize indicator data
        if self.indicator_model:
            input_data = self._synthesize_indicator_data(input_data)
            if input_data.empty:
                logging.debug(f'synthesize model {self._feature_name} indicator empty input data.')
                return output

        if self._has_data:
            if self._endpoints:
                # then update values for target feature,
                # for the indices in the synthesize-data corresponding to given leaf id
                has_weights = bool(self._leaf_ids_to_endpoints_weights)
                leaf_ids = self._endpoints_tree.apply(input_data.to_numpy())
                input_data['Leaf_ID'] = leaf_ids  # Add leaf ids
                for leaf_id, group in input_data.groupby('Leaf_ID'):
                    # using the mapping of leaf ids to real target values, sample with replacement
                    # for the number of rows (from synthesized data, given trained tree) with that leaf id
                    sampled_values = choices(self._leaf_ids_to_endpoints_values[leaf_id],
                                             weights=self._leaf_ids_to_endpoints_weights[leaf_id]
                                             if has_weights else None,
                                             k=group.shape[0])

                    # update values of indices corresponding to curr leaf id
                    indices_curr_leaf_id = group.index.tolist()
                    cur_output = self.output_series(data=sampled_values, indexes=indices_curr_leaf_id)
                    output.update(cur_output)
                input_data.drop(labels=['Leaf_ID'], axis=1, inplace=True)
                input_data = input_data.loc[~output.isin(self._endpoints), :].copy()
        if self._has_data:
            # then update values for target feature,
            # for the indices in the synthesize-data corresponding to given leaf id
            has_weights = bool(self._leaf_ids_to_target_weights)
            leaf_ids = self.tree.apply(input_data.to_numpy())
            input_data['Leaf_ID'] = leaf_ids  # Add leaf ids
            for leaf_id, group in input_data.groupby('Leaf_ID'):
                # using the mapping of leaf ids to real target values, sample with replacement for the number of rows
                # (from synthesized data, given trained tree) with that leaf id
                sampled_values = choices(self._leaf_ids_to_target_feature_values[leaf_id],
                                         weights=self._leaf_ids_to_target_weights[leaf_id] if has_weights else None,
                                         k=group.shape[0])

                # update values of indices corresponding to curr leaf id
                indices_curr_leaf_id = group.index.tolist()
                cur_output = self.output_series(data=sampled_values, indexes=indices_curr_leaf_id)
                output.update(cur_output)

        return output


class DecisionTreeRegressorModel(DecisionTreeBaseModel):

    def __init__(self, *args, **kwargs) -> None:
        """
        Init for DecisionTreeRegressorModel

        :param: args: Positional arguments passed on to super.
        :param: kwargs: Keyword arguments passed on to super.
        """
        super().__init__(*args, **kwargs)

        if self._target_feature.dependencies is None:
            raise RuntimeError('Attempted initialization of a DecisionTreeRegressorModel with no dependencies.')
        logging.debug(msg=f"Instantiate DecisionTreeModel for {self._feature_name}.")

    def validate_sklearn_params(self, sklearn_params: Dict = None) -> Dict:
        """
        Validate the sklearn model parameters for the decision tree.

        :param: sklearn_params: Dictionary of sklearn parameters.
        :return: Valid dictionary of sklearn parameters.
        """
        logging.debug(msg=f"DecisionTreeRegressorModel.validate_sklearn_params() for {self._feature_name}.")
        if sklearn_params is None:
            sklearn_params = {'max_depth': 10, 'criterion': 'squared_error', 'min_impurity_decrease': 1e-5}
        return sklearn_params

    def get_decision_tree(self, sklearn_params: Dict) -> TreeRegressor:
        """
        Generate the decision tree for the model

        :param: sklearn_params:  Dictionary of sklearn parameters.
        :return: Decision Tree
        """
        logging.debug(msg=f"DecisionTreeRegressorModel.get_decision_tree() for {self._feature_name}.")
        return TreeRegressor(**sklearn_params)

    def _not_endpoint(self) -> Union[float, None]:
        """
        Get value which is not in the endpoints. This is utilized in defining the target data.

        :return: A value not in the endpoints.
        """
        test_value = 0.0
        while test_value in self._endpoints:
            test_value += 1
        return test_value

    def synthesize(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Synthesizes the values for a feature. The number of rows synthesized is informed by the shape of input_data.
        This method checks to see if the features specified in dependencies have been synthesized.
        On successful completion of a call to synthesize(), the synthesized property is set to True.

        :param: input_data: pd.DataFrame containing the features (encoded) that this model is dependant upon.
        :return: pd.Series of synthesized values (unencoded) for target feature.
        """
        logging.debug(msg=f"DecisionTreeRegressorModel.synthesize() for {self._feature_name}. "
                          f"input_data {input_data.shape}.")
        self.validate_trained()
        self.validate_dependencies(input_data)

        output = self.output_series(data=None, indexes=input_data.index)

        # Synthesize indicator data
        if self.indicator_model:
            input_data = self._synthesize_indicator_data(input_data)
            if input_data.empty:
                logging.debug(f'synthesize model {self._feature_name} indicator empty input data.')
                return output

        if self._has_data:
            synth_values = self.tree.predict(input_data.to_numpy())
            output = self.output_series(data=synth_values, indexes=input_data.index)

            if self._output_encoded and self._category is not None:
                output = pd.Categorical.from_codes(codes=output, dtype=self._category.dtype).tolist()
                output = self.output_series(data=output, indexes=input_data.index)
        return output
