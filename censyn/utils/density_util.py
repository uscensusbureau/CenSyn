import logging
from statistics import mean
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def compute_density_mean(density_to_compute: List[Tuple[Tuple, float]]) -> float:
    """
    Calculates the mean of the absolute differences.

    :param: density_to_compute: A list of tuples of feature combinations and absolute densities differences.
    :return: A single mean of the density_to_compute scores.
    """
    logging.debug(msg=f"density_util.compute_density_mean().")
    return mean([density[1] for density in density_to_compute])


def compute_density_total_difference(single_density_diff: pd.DataFrame) -> float:
    """
    Finds the total difference in a density's values.

    :param: single_density_diff: Tuple of the combination and density differences.
    :return: A tuple of the combination and summed difference of the density difference.
    """
    # logging.debug(msg=f"density_util.compute_density_total_difference(). "
    #                   f"single_density_diff {single_density_diff.shape}.")
    summed_diff: float = single_density_diff[single_density_diff.columns[0]].sum()
    return summed_diff


def compute_density_distribution_differences(density_distribution_differences: Dict) -> List[Tuple[Tuple, float]]:
    """
    Computes and returns the absolute differences of the densities between the data sets.
    In other words: Abs_Diff( Density Distributions )
    This means computing the N-marginal score (sum of abs diff for two dens dist.) for each N-marginal.

    :param: density_distribution_differences: Dictionary of feature combination and density distribution differences.
    :return: List of absolute differences of the densities between the data.
    """
    logging.debug(msg=f"density_util.compute_density_distribution_differences().")
    results = [(diff_key, compute_density_total_difference(diff_value))
               for diff_key, diff_value in density_distribution_differences.items()]
    return results


def compute_density_difference(feature_combination: Tuple, density_a: pd.DataFrame,
                               density_b: pd.DataFrame) -> Tuple[Tuple, pd.DataFrame]:
    """
    Finds and returns the absolute difference between two densities.

    :param: feature_combination: A tuple of the features to process.
    :param: density_a: The density of the current feature combination from the dataset_a.
    :param: density_b: The density of the current feature combination from the dataset_b.
    :returns: Tuple of the combination and density differences.
    """
    # logging.debug(msg=f"density_util.compute_density_difference(). feature_combination {feature_combination}")
    merged_densities = density_a.join(density_b, how='outer', lsuffix='_A', rsuffix='_B').fillna(0)
    density_distribution_difference = (merged_densities[merged_densities.columns[0]] -
                                       merged_densities[merged_densities.columns[1]]).abs()
    merged_densities.insert(loc=0, column='diff', value=density_distribution_difference)
    return feature_combination, merged_densities


def compute_feature_combinations_to_density_distributions(feature_combination: Tuple,
                                                          in_df: pd.DataFrame,
                                                          in_weight: Union[pd.Series, None]) -> Tuple[Tuple,
                                                                                                      pd.DataFrame]:
    """
    Given a feature combination and two dataframes, return density_distributions for that combination.
    This was specifically made to support multiprocessing.

    :param: feature_combination: A tuple of the features to process
    :param: in_df: The first DataFrame
    :param: in_weight: weight of the first of two data frames
    :returns: Tuple of the combination and density differences.
    """
    logging.debug(msg=f"density_util.compute_feature_combinations_to_density_distributions().")
    features = list(feature_combination)
    if in_weight is not None:
        number_rows = np.sum(in_weight)
        in_df[in_weight.name] = in_weight
        n_marginal_sizes = in_df.groupby(features)[in_weight.name].sum()
    else:
        number_rows = in_df.shape[0]
        n_marginal_sizes = in_df.groupby(features).size()

    density_distribution = pd.DataFrame(data=n_marginal_sizes / number_rows, index=n_marginal_sizes.index)
    return feature_combination, density_distribution


def compute_mapping_feature_combinations_to_density_distribution_differences(
        feature_combinations_to_density_distribution_a,
        feature_combinations_to_density_distribution_b) -> Dict:
    """
    Compute the absolute difference of density distributions for two data frames, for each feature combination.

    :return: Dictionary of form {n-feature-combo : [density distribution]}
    """
    feature_combinations_to_absolute_density_difference = {}
    for (feature_combination_ab, density_distribution_a), (feature_combination_ab, density_distribution_b) in zip(
            feature_combinations_to_density_distribution_a.items(),
            feature_combinations_to_density_distribution_b.items()):
        res = compute_density_difference(feature_combination_ab, density_distribution_a, density_distribution_b)
        feature_combinations_to_absolute_density_difference[res[0]] = res[1]

    return feature_combinations_to_absolute_density_difference
