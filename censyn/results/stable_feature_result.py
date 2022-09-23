from typing import List, Dict

import pandas as pd

from .table_result import TableResult
from censyn.utils.stable_features import StableFeatures


class StableFeatureResult(TableResult):
    """Result subclass for StableFeature objects"""

    def __init__(self, *args, sf: List, data_a_count: int, data_b_count: int, baseline_count: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display_number_lines = 0

        self._sf = sf
        self._data_a_count = data_a_count
        self._data_b_count = data_b_count
        self._baseline_count = baseline_count
        self._display_minimum_data_count = 0
        self._display_minimum_percent = 100.0
        self._display_density = False
        self.title = f'Stable Feature{"s" if len(sf) > 1 else ""} Marginals Values'
        self.value.insert(1, StableFeatures.DATA_A_COUNT,
                          self.value[StableFeatures.DENSITY_A].apply(
                              lambda x: round(x * self._data_a_count)).astype(int))
        self.value.insert(2, StableFeatures.DATA_B_COUNT,
                          self.value[StableFeatures.DENSITY_B].apply(
                              lambda x: round(x * self._data_b_count)).astype(int))
        if baseline_count:
            self.value.insert(3, StableFeatures.BASELINE_COUNT,
                              self.value[StableFeatures.DENSITY_BASELINE].apply(
                                  lambda x: round(x * self._baseline_count)).astype(int))

    @property
    def display_minimum_percent(self) -> float:
        """display_minimum_percent of the report property."""
        return self._display_minimum_percent

    @display_minimum_percent.setter
    def display_minimum_percent(self, value: float) -> None:
        """Setter for the display_minimum_percent of the report property."""
        self._display_minimum_percent = value

    @property
    def display_minimum_data_count(self) -> int:
        """stable_feature minimum_data_count of the report property."""
        return self._display_minimum_data_count

    @display_minimum_data_count.setter
    def display_minimum_data_count(self, value: int) -> None:
        """Setter for the stable_feature minimum_data_count of the report property."""
        self._display_minimum_data_count = value

    @property
    def display_density(self) -> bool:
        """stable_feature display_density of the report property."""
        return self._display_density

    @display_density.setter
    def display_density(self, value: bool) -> None:
        """Setter for the stable_feature_display_density of the report property."""
        self._display_density = value

    @staticmethod
    def _set_minimum_percent(display_df: pd.DataFrame, percent: float) -> pd.DataFrame:
        """
        Modify a DataFrame to display the minimum data count of data set a for the rows. This eliminates the data
        with very small percentage counts which may just be outliers.

        :param: display_df: DataFrame to perform upon.
        :param: percent: Percentage of data to display.
        :return: DataFrame with the minimum data count for the rows.
        """
        # Remove the lowest density rows as needed.
        if percent < 100.0:
            count_df = display_df.sort_values(by=StableFeatures.DATA_A_COUNT, axis=0, ascending=False, inplace=False)
            sum_count = count_df[StableFeatures.DATA_A_COUNT].sum()
            min_count = round(sum_count * percent / 100)
            for i in range(count_df.shape[0]):
                min_count = min_count - count_df.at[i, StableFeatures.DATA_A_COUNT]
                if min_count <= 0:
                    display_df.loc(count_df.head(i + 1).index)
                    break
        return display_df

    @staticmethod
    def _set_minimum_data_count(df: pd.DataFrame, count: int) -> pd.DataFrame:
        """
        Modify a DataFrame to display the minimum data count for the rows. This eliminates the data with small counts
        which may just be outliers.

        :param: df: DataFrame to perform upon.
        :param: count: the required minimum count from both 'Data_a Count' and 'Data_b Count' columns.
        :return: DataFrame with the minimum data count for the rows.
        """
        if count > 0:
            df = df[(df[StableFeatures.DATA_A_COUNT] >= count) | (df[StableFeatures.DATA_B_COUNT] >= count)]
        return df

    def display_bins_readable_values(self, features: Dict):
        """
        Modify the value DataFrame to display the readable values for the stable features' bins.

        :param: features: Dictionary of Features.
        :return: None
        """
        if len(self._sf) <= 1:
            for feature_name in self._sf:
                for f_name, feat in features.items():
                    if f_name == feature_name:
                        if feat.binner and feat.binner.bin_list is not None and len(feat.binner.bin_list) > 0:
                            bins = {i: feat.binner.bin_list[i][0] for i in range(len(feat.binner.bin_list))}
                            self.value.replace({'Bins': bins}, inplace=True)
                        break
        else:
            split_df = pd.DataFrame(self.value[StableFeatures.BINS].to_list(), index=self.value.index)
            level = 0
            for feature_name in self._sf:
                for f_name, feat in features.items():
                    if f_name == feature_name:
                        if feat.binner and feat.binner.bin_list is not None and len(feat.binner.bin_list) > 0:
                            bins = {i: feat.binner.bin_list[i][0] for i in range(len(feat.binner.bin_list))}
                            split_df.replace({level: bins}, inplace=True)
                        level = level + 1
                        break
            self.value['Bins'] = split_df[:].apply(tuple, axis=1)

    def _set_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the display columns for the result.

        :param: df:  DataFrame to perform upon.
        :return: DataFrame with the columns for display.
        """
        if not self.display_density:
            df.drop(columns=[StableFeatures.DENSITY_A, StableFeatures.DENSITY_B], axis=1, inplace=True)
            if self._baseline_count:
                df.drop(columns=[StableFeatures.DENSITY_BASELINE], axis=1, inplace=True)

        return df

    def display_value(self) -> str:
        """The display string for the result's value."""
        display_df = self._set_minimum_percent(self.value.copy(), self.display_minimum_percent)
        display_df = self._set_minimum_data_count(display_df, self.display_minimum_data_count)
        display_df = self._set_display_columns(display_df)

        # Sort if the column is defined.
        display_df = self.sort_display(display_df)

        pt = self.create_pretty_table(display_df)
        pt.align['Bins'] = 'l'
        if self.display_density:
            density_format = self.calc_float_format(display_df[StableFeatures.DENSITY_A])
            density_format_b = self.calc_float_format(display_df[StableFeatures.DENSITY_B])
            if density_format_b > density_format:
                density_format = density_format_b
            pt.float_format[StableFeatures.DENSITY_A] = density_format
            pt.float_format[StableFeatures.DENSITY_B] = density_format
            if self._baseline_count:
                pt.float_format[StableFeatures.DENSITY_BASELINE] = density_format
        if self._baseline_count:
            pt.float_format[StableFeatures.BASELINE_RAW_SCORE] = \
                self.calc_float_format(display_df[StableFeatures.BASELINE_RAW_SCORE])
            pt.float_format[StableFeatures.BASELINE_NIST_SCORE] = '4.3'
        pt.float_format[StableFeatures.RAW_SCORE] = self.calc_float_format(display_df[StableFeatures.RAW_SCORE])
        pt.float_format[StableFeatures.NIST_SCORE] = '4.3'
        output: str = '\n'
        output += str(pt)
        return output
