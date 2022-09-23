import logging
from typing import List, Union

import pandas as pd

from censyn.features import FeatureType
from censyn.metrics import Metrics, MarginalMetric
from ..results.result import Result, ListResult, MappingResult


SUFFIX_A = "_A"
SUFFIX_B = "_B"


class JoinMarginalMetric(MarginalMetric):

    def __init__(self, join_features: List[str], edge_features: List[str], edge_values: List[str] = None,
                 duplicate_features: List[str] = None, retain_duplicates: bool = False, *args, **kwargs) -> None:
        """
        Initialization for JoinMarginalMetric

        :param: join_features: List of join features
        :param: edge_features: List of edge features
        :param: edge_values: List of edge values.
        :param: duplicate_features: List of features to uniquely identify rows for duplicate elimination.
        :param: retain_duplicates: Keep duplicate record
        :param: args:
        :param: kwargs:
        """
        self._duplicate_features = duplicate_features if duplicate_features else []
        if "duplicate_feature" in kwargs.keys():
            duplicate_feature = kwargs["duplicate_feature"]
            del kwargs["duplicate_feature"]
            if duplicate_feature not in self._duplicate_features:
                self._duplicate_features.append(duplicate_feature)
            logging.warning(f"Deprecated parameter duplicate_feature. Use duplicate_features instead.")
        super().__init__(*args, **kwargs)
        logging.debug(msg=f"Instantiate JoinMarginalMetric with join_features {join_features}, "
                          f"edge_features {edge_features}, edge_values {edge_values}, "
                          f"duplicate_features {duplicate_features} and retain_duplicates {retain_duplicates}.")

        self._join_features = join_features
        self._edge_features = edge_features
        # When no edge values add the empty string to force processing of an edge
        self._edge_values = edge_values if edge_values else [""]
        self._retain_duplicates = retain_duplicates

        # Save the original stable features
        self._original_sf = self._stable_features

    @Metrics.use_bins.getter
    def use_bins(self) -> bool:
        """
        Getter for the use bins flag. The JoinMarginalMetric needs to have the original non binned values
        to enable the use of data values. This make sire the join_features have not been modifies to
        invalidate the results.
        """
        return False

    @property
    def in_process_features(self) -> List[str]:
        """Getter for in process features."""
        return self._edge_features

    @property
    def additional_features(self) -> List[str]:
        """Getter for additional features."""
        add_features = self._join_features + self._duplicate_features
        return add_features

    def validate_data(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                      weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                      add_a: Union[pd.DataFrame, None], add_b: Union[pd.DataFrame, None]):
        """
        Check that the data sets are valid.

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        """
        super().validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

        if add_a is None or add_b is None:
            if self._join_features:
                msg = f"Additional data not set. Join features {self._join_features} have to exists in the data."
                logging.error(msg=msg)
                raise ValueError(msg)
            if self._duplicate_features:
                msg = f"Additional data not set. Duplicate feature {self._duplicate_features} has " \
                      f"to exists in the data."
                logging.error(msg=msg)
                raise ValueError(msg)
        else:
            if self._join_features and not all((f_name in add_a.columns for f_name in self._join_features)):
                msg = f"Join features {self._join_features} have to exists in the data."
                logging.error(msg=msg)
                raise ValueError(msg)
            for feat in self._duplicate_features:
                if feat not in add_a.columns:
                    msg = f"Duplicate feature {feat} has to exists in the data."
                    logging.error(msg=msg)
                    raise ValueError(msg)
        if self._edge_features and not all((f_name in data_frame_a.columns for f_name in self._edge_features)):
            msg = f"Edge features {self._edge_features} have to exists in the data."
            logging.error(msg=msg)
            raise ValueError(msg)

    def compute_results(self, data_frame_a: pd.DataFrame, data_frame_b: pd.DataFrame,
                        weight_a: Union[pd.Series, None], weight_b: Union[pd.Series, None],
                        add_a: Union[pd.DataFrame, None] = None, add_b: Union[pd.DataFrame, None] = None) -> Result:
        """
        Calculate marginal metric results from the joining of the data.

        :param: data_frame_a: First of data frames to evaluate
        :param: data_frame_b: Second of data frames to evaluate
        :param: weight_a: weight of the first of two data frames to evaluate
        :param: weight_b: weight of the second of two data frames to evaluate
        :param: add_a: Data frames for additional first data set
        :param: add_b: Data frames for additional second data set
        :return: one Result obj, containing all component Result objs
        """
        self.validate_data(data_frame_a, data_frame_b, weight_a, weight_b, add_a=add_a, add_b=add_b)

        # Join additional columns data to the process data.
        join_a = [col for col in add_a.columns if col not in data_frame_a.columns] if add_a is not None else []
        data_frame_a = data_frame_a.join(add_a[join_a])
        join_b = [col for col in add_b.columns if col not in data_frame_b.columns] if add_b is not None else []
        data_frame_b = data_frame_b.join(add_b[join_b])

        # Initialize the empty output results.
        results = []
        weight_name = weight_a.name if weight_a is not None else ""

        # Process each edge
        for edge in self._edge_values:
            dfa0 = data_frame_a.copy()
            dfa1 = data_frame_a.copy()
            dfb0 = data_frame_b.copy()
            dfb1 = data_frame_b.copy()

            if weight_name:
                dfa0[weight_name] = weight_a
                dfa1[weight_name] = weight_a
                dfb0[weight_name] = weight_b
                dfb1[weight_name] = weight_b

            # Filter the data sets with the edge values
            dfa0, dfa1, dfb0, dfb1 = self._filter_edges(edge, dfa0, dfa1, dfb0, dfb1)

            # Join the data set together.
            dfa = self._join_data_set(left_df=dfa0, right_df=dfa1)
            logging.info(f'Merge Data A size {dfa.shape[0]} rows by {dfa.shape[1]} columns')
            dfb = self._join_data_set(left_df=dfb0, right_df=dfb1)
            logging.info(f'Merge Data B size {dfb.shape[0]} rows by {dfb.shape[1]} columns')

            if self._duplicate_features:
                logging.info(f'Process duplicate features {self._duplicate_features}')
                dfa = self._process_duplicates(in_df=dfa, data_name="Data A")
                dfb = self._process_duplicates(in_df=dfb, data_name="Data B")

            weight_a = self._calculate_weight(weight_name, dfa)
            weight_b = self._calculate_weight(weight_name, dfb)

            # Modify the stable features for the join data
            # sets.
            self._modify_stable_features(edge)

            # Drop the joined additional data from the process data
            self._drop_columns(df=dfa, cols=join_a)
            self._drop_columns(df=dfb, cols=join_b)

            dfa = self._transform_bin_data(dfa)
            dfb = self._transform_bin_data(dfb)

            cur_results = super().compute_results(dfa, dfb, weight_a=weight_a, weight_b=weight_b)
            self._update_result(cur_results, edge=edge)
            results.append(cur_results)

        # Create ListResults using all the k-marginal metric results.
        to_return = ListResult(value=results, metric_name=self.name,
                               description='All of the results for the join marginal metric computation')
        return to_return

    @staticmethod
    def _drop_columns(df: pd.DataFrame, cols: List[str]) -> None:
        """
        Drop the columns from the DataFrame.

        :param: df: Pandas data frame.
        :param: cols: List of column names.
        :return: None
        """
        for col in cols:
            if col in df.columns:
                df.drop(col, axis='columns', inplace=True)
            else:
                df.drop(col + SUFFIX_A, axis='columns', inplace=True)
                df.drop(col + SUFFIX_B, axis='columns', inplace=True)

    def _filter_edges(self, edge: str, dfa0: pd.DataFrame, dfa1: pd.DataFrame, dfb0: pd.DataFrame, dfb1: pd.DataFrame):
        """
        Filter the data sets based on the edge values for each part.

        :param edge: The current edge.
        :param dfa0: Data Set A left side.
        :param dfa1: Data Set A right side.
        :param dfb0: Data Set B left side.
        :param dfb1: Data Set B right side.
        :return:
        """
        if edge:
            parts = edge.split(sep=':')
            assert len(parts) == 2
            edge_values_0 = parts[0].split(sep=',')
            edge_values_1 = parts[1].split(sep=',')
            assert len(edge_values_0) == len(self._edge_features)
            assert len(edge_values_1) == len(self._edge_features)

            index = 0
            for edge_f in self._edge_features:
                p0 = edge_values_0[index]
                p1 = edge_values_1[index]
                if self.features:
                    if self.features[edge_f].feature_type == FeatureType.integer:
                        p0 = int(p0)
                        p1 = int(p1)
                    elif self.features[edge_f].feature_type == FeatureType.floating_point:
                        p0 = float(p0)
                        p1 = float(p1)
                dfa0 = dfa0[dfa0[edge_f] == p0]
                dfa1 = dfa1[dfa1[edge_f] == p1]
                dfb0 = dfb0[dfb0[edge_f] == p0]
                dfb1 = dfb1[dfb1[edge_f] == p1]
                index += 1

        return dfa0, dfa1, dfb0, dfb1

    def _modify_stable_features(self, edge: str):
        """
        Modify the stable features for the join data sets which uses the SUFFIX.

        :param: edge: The current edge
        """
        mod_edge_features = []
        if not edge:
            for feat in self._edge_features:
                mod_edge_features.append(feat + SUFFIX_A)
                mod_edge_features.append(feat + SUFFIX_B)

        mod_stable_features = []
        if self._original_sf:
            for feat in self._original_sf:
                mod_stable_features.append(feat + SUFFIX_A)
                mod_stable_features.append(feat + SUFFIX_B)
        self._stable_features = mod_edge_features
        self._stable_features.extend(mod_stable_features)

    def _join_data_set(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the left and right data sets on join_features.

        :param left_df: Left Pandas data frame.
        :param right_df: Right Pandas data frame.
        :return: Joined Pandas data frame.
        """
        out_df = left_df.merge(right=right_df, how="inner", left_on=self._join_features,
                               right_on=self._join_features, suffixes=(SUFFIX_A, SUFFIX_B))
        return out_df

    def _process_duplicates(self, in_df: pd.DataFrame, data_name: str = "Data") -> pd.DataFrame:
        """
        Remove duplicate entries from the joined data frame.

        :param: in_df: Joined Pandas data frame.
        :param: data_name: String name for the dataset.
        :return: Pandas data frame.
        """
        if self._duplicate_features:
            equal_mask = pd.Series(data=True, index=in_df.index)
            less_mask = pd.Series(data=False, index=in_df.index)
            for feat in self._duplicate_features:
                if feat in in_df.columns:
                    feat_a = feat
                    feat_b = feat
                else:
                    feat_a = feat + SUFFIX_A
                    feat_b = feat + SUFFIX_B
                less_mask = less_mask | (equal_mask & (in_df[feat_a] < in_df[feat_b]))
                equal_mask = equal_mask & (in_df[feat_a] == in_df[feat_b])
            if self._retain_duplicates:
                out_df = in_df[~equal_mask]
            else:
                out_df = in_df[less_mask]
        else:
            out_df = in_df
        logging.info(f'Merge {data_name} size {out_df.shape[0]} rows by {out_df.shape[1]} columns')
        return out_df

    def _transform_bin_data(self, in_df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin all the features' data.

        :param: in_df: The Pandas DataFrame of the  data.
        :return: Binned DataFrame.
        """
        if super().use_bins:
            logging.info(f'Transforming {len(in_df.columns)} features')
            if not self.features:
                logging.error(msg=f"Features not defined for binning.")
                raise ValueError(f"Features not defined for binning.")
            for col in in_df.columns:
                binner = self.features[self.get_feature_name(col)].binner
                if binner is not None:
                    in_df[col] = binner.bin(in_s=in_df[col])

        return in_df

    @staticmethod
    def _calculate_weight(weight_name: str, in_df: pd.DataFrame) -> Union[pd.Series, None]:
        """
        Calculate the weight for the data.

        :param: weight_name: The original name of the feature weight without suffixes.
        :param: in_df: The data set with the weight features.
        :return: Pandas Series of weights.
        """
        if weight_name:
            weight_s = in_df[weight_name + SUFFIX_A].mul(in_df[weight_name + SUFFIX_B])
            in_df.drop(columns=[weight_name + SUFFIX_A, weight_name + SUFFIX_B], inplace=True)
            return weight_s
        return None

    def _update_result(self, result: Result, edge: str) -> None:
        """
        Appends the result to the report.

        :param: result: Result to append to the report.
        :param: edge: the edge values used for computing the marginal metric.
        :return: None
        """
        if result.container:
            if isinstance(result.value, List):
                for res in iter(result.value):
                    if isinstance(res, MappingResult):
                        if res.description.startswith('Summary:\n'):
                            name = f"Join marginal metric"
                            ef_str = ""
                            if len(self._edge_features) > 0:
                                name = f"{name} with edges "
                                for ef in self._edge_features:
                                    if ef_str:
                                        ef_str = f"{ef_str}, {ef}"
                                    else:
                                        ef_str = ef
                                name = f"{name}{ef_str}"
                                if edge:
                                    parts = edge.split(':')
                                    if len(parts) == 2:
                                        left_str = ""
                                        right_str = ""
                                        left_edge = parts[0].split(',')
                                        right_edge = parts[1].split(',')
                                        for i in range(len(self._edge_features)):
                                            feat = self.features[self._edge_features[i]]
                                            feat_bin = feat.binner
                                            edge_s = pd.Series(data=[left_edge[i].strip(), right_edge[i].strip()])
                                            edge_s = feat.set_series_data_type(in_s=edge_s)
                                            bin_edges = feat_bin.bin(edge_s)
                                            bin_labels = feat_bin.bins_to_labels(bin_edges)
                                            left_str = f"{left_str}{', ' if left_str else ''}{bin_labels[0]}"
                                            right_str = f"{right_str}{', ' if right_str else ''}{bin_labels[1]}"
                                        name = f"{name} with values of {left_str} : {right_str}"
                            res.metric_name = name
                            return
                    self._update_result(res, edge=edge)

    def get_feature_name(self, col: str) -> str:
        """
        Get the feature name associated with a column name.

        :param: col: Name of the column data.
        :return: Feature name
        """
        if self.features:
            if col in self.features:
                return col
            elif col.endswith(SUFFIX_A) and col[:-len(SUFFIX_A)] in self.features:
                return col[:-len(SUFFIX_A)]
            elif col.endswith(SUFFIX_B) and col[:-len(SUFFIX_B)] in self.features:
                return col[:-len(SUFFIX_B)]
            msg = f"No feature associated with column {col}."
            logging.error(msg=msg)
            raise ValueError(msg)
        if not self.data_frame_a.empty:
            if col.endswith(SUFFIX_A) and col[:-len(SUFFIX_A)] in self.data_frame_a.columns:
                return col[:-len(SUFFIX_A)]
            elif col.endswith(SUFFIX_B) and col[:-len(SUFFIX_B)] in self.data_frame_a.columns:
                return col[:-len(SUFFIX_B)]
            elif col in self.data_frame_a.columns:
                return col
        msg = f"No feature associated with column {col}."
        logging.error(msg=msg)
        raise ValueError(msg)
