import logging
from enum import Enum
from statistics import mean
from typing import Union

import pandas as pd


class StableFeaturesColumn(Enum):
    """
    This is an enumeration that represents the columns of the stable feature
    """
    bins = 0
    data_a_count = 1
    data_b_count = 2
    nist = 3


class StableFeatures:
    # Column names
    BINS: str = "Bins"
    DATA_A_COUNT = 'Data_a Count'
    DATA_B_COUNT = 'Data_b Count'
    BASELINE_COUNT = 'Baseline Count'
    DENSITY_A: str = "Density a"
    DENSITY_B: str = "Density b"
    DENSITY_BASELINE: str = "Density Baseline"
    RAW_SCORE: str = 'Raw score'
    NIST_SCORE: str = 'NIST score'
    BASELINE_RAW_SCORE: str = 'Baseline Raw score'
    BASELINE_NIST_SCORE: str = 'Baseline NIST score'

    def __init__(self, names: [], sort_col: StableFeaturesColumn = StableFeaturesColumn.nist,
                 ascending: bool = True) -> None:
        """
        The stable features marginal scores

        :param: names: The stable features
        :param: sort_col: The column on which the table is sorted. Default is nist.
        :param: ascending: Boolean if the sort is ascending or descending. Default is True.
        """
        logging.debug(msg=f"Instantiate StableFeatures with names {names}, sort_col {sort_col} and "
                          f"ascending {ascending}.")
        self._names = names
        self._scores = {}
        self._baseline_scores = {}   # Scores keyed by stable feature bin number
        self._density_a = {}
        self._density_b = {}
        self._baseline_density = {}
        self._count = 0
        self._dirty = False
        self._sort_col = sort_col
        self._ascending = ascending

    def add_scores(self, score_df: pd.DataFrame, baseline_score_df: Union[pd.DataFrame, None] = None) -> None:
        """
        Add marginal scores for the set of features grouped by the stable feature.

        :param: score_df: Data frame of marginal scores.
        :param: baseline_score_df: Data frame of the baseline marginal scores
        """
        logging.debug(msg=f"StableFeatures.add_scores(score_df={score_df.shape}, "
                          f"baseline_score_df={'None' if baseline_score_df is None else baseline_score_df.shape}).")
        top = score_df.groupby(level=self._names).sum()

        # Accumulate stable marginals densities and scores
        for sv in list(top.index):
            sv_top = top.loc[sv]
            self._density_a[sv] = self._density_a.get(sv, 0) + sv_top[1]
            self._density_b[sv] = self._density_b.get(sv, 0) + sv_top[2]
            self._scores[sv] = self._scores.get(sv, [])
            sv_sum = sv_top[1] + sv_top[2]
            self._scores[sv].append(0.0 if sv_sum == 0.0 else sv_top[0] * 2 / sv_sum)

        if baseline_score_df is not None:
            baseline_top = baseline_score_df.groupby(level=self._names).sum()
            for sv in list(baseline_top.index):
                sv_top = baseline_top.loc[sv]
                self._baseline_density[sv] = self._baseline_density.get(sv, 0) + sv_top[2]
                self._baseline_scores[sv] = self._baseline_scores.get(sv, [])
                sv_sum = sv_top[1] + sv_top[2]
                self._baseline_scores[sv].append(0.0 if sv_sum == 0.0 else sv_top[0] * 2 / sv_sum)

        self._count = self._count + 1
        self._dirty = True

    def scores(self) -> pd.DataFrame:
        """The Stable Features marginal score."""
        logging.debug(msg=f"StableFeatures.scores().")
        self._clean()

        # pretty it up for display
        keys = [k for k in self._scores.keys()]
        bin_s = pd.Series(data=keys, name=StableFeatures.BINS)
        df = pd.DataFrame(data=bin_s, columns=[StableFeatures.BINS])
        df[StableFeatures.DENSITY_A] = self._density_a.values()
        df[StableFeatures.DENSITY_A] = df[StableFeatures.DENSITY_A].apply(lambda x: x / self._count)
        df[StableFeatures.DENSITY_B] = self._density_b.values()
        df[StableFeatures.DENSITY_B] = df[StableFeatures.DENSITY_B].apply(lambda x: x / self._count)
        if self._baseline_scores:
            df[StableFeatures.DENSITY_BASELINE] = self._baseline_density.values()
            df[StableFeatures.DENSITY_BASELINE] = df[StableFeatures.DENSITY_BASELINE].apply(lambda x: x / self._count)
            df[StableFeatures.BASELINE_RAW_SCORE] = self._baseline_scores.values()
            df[StableFeatures.BASELINE_NIST_SCORE] = self._baseline_scores.values()
            df[StableFeatures.BASELINE_NIST_SCORE] = df[StableFeatures.BASELINE_NIST_SCORE].apply(
                lambda x: ((2 - x) / 2) * 1000)

        df[StableFeatures.RAW_SCORE] = self._scores.values()
        df[StableFeatures.NIST_SCORE] = self._scores.values()
        df[StableFeatures.NIST_SCORE] = df[StableFeatures.NIST_SCORE].apply(lambda x: ((2 - x) / 2) * 1000)

        if self._sort_col == StableFeaturesColumn.bins:
            return df.sort_values(by=StableFeatures.BINS, axis=0, ascending=self._ascending)
        if self._sort_col == StableFeaturesColumn.data_a_count:
            return df.sort_values(by=StableFeatures.DENSITY_A, axis=0, ascending=self._ascending)
        if self._sort_col == StableFeaturesColumn.data_b_count:
            return df.sort_values(by=StableFeatures.DENSITY_B, axis=0, ascending=self._ascending)
        if self._sort_col == StableFeaturesColumn.nist:
            return df.sort_values(by=StableFeatures.NIST_SCORE, axis=0, ascending=self._ascending)
        return df.sort_values(by=StableFeatures.RAW_SCORE, axis=0, ascending=self._ascending)

    def _clean(self) -> None:
        """Perform addition tasks for returned data."""
        if self._dirty:
            # Divide scores by the count within the combined set of stable feature marginals
            for k in self._scores.keys():
                self._scores[k] = mean(self._scores[k])
            for k in self._baseline_scores.keys():
                self._baseline_scores[k] = mean(self._baseline_scores[k])
            self._dirty = False
