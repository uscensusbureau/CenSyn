import concurrent.futures.process
import itertools
import math
import os
from multiprocessing import Pool
from typing import Any, List, Callable, Generator

import pandas as pd
from bounded_pool_executor import BoundedProcessPoolExecutor
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


def pool_helper(process_num: int, functions: List[Callable]):
    """
    In general this will start all the threads with the provided cap and return the results from the provided functions.

    :param process_num: The number of cores to run on.
    :param functions: This is a list of functions that are run on a Process Pool.
        Example [partial(process_function, feature=feature, series_a=df_a[feature.feature_name],
        series_b=df_b[feature.feature_name])]
    """
    futures = []
    with Pool(process_num) as pool:
        for function in functions:
            futures.append(pool.apply_async(function))
        return blocking_pool(futures)


def pool_map_helper(process_num: int, function: List[Callable], to_run_combinations):
    """
    Call all functions and map the to_run_combinations to each of the started processes.

    :param process_num: The number of cores to run on.
    :param function: a list of functions that are run on a Process Pool.
    :param to_run_combinations: a list of feature combinations that get mapped out to the different processes.
    :return: A list of results from the called functions.
    """
    with Pool(process_num) as pool:
        results = pool.map(function, to_run_combinations)
    return results


def bounded_pool(process_num, functions: Generator, process_factor: int = 3) -> List:
    """
    In general this will start all the threads with the provided cap. If not blocking, it will return the features;
    otherwise it will return results.

    To facilitate better pre-processing because functions can be a generator, this uses  a process_factor of three to
    pre-process three times the size of the bounded pool. Further, when this list gets smaller than the worker size this
    adds more pre-processed functions to the list. When the function is memory intensive and memory is more of a
    consideration the function can be a generator and the process_factor could be set to 1.

    :param: process_num: The number of cores to run on.
    :param: functions: A list of functions.
        Example [partial(process_function,
                         feature=feature,
                         series_a=df_a[feature.feature_name],
                         series_b=df_b[feature.feature_name])]
    :param: process_factor: process stack buffer size a factor of process_num. Default is 3.
    :return: List of whatever is produced by the processes.
    """
    futures = []
    process_size = process_num * process_factor
    process_stack = list(itertools.islice(functions, process_size))
    with BoundedProcessPoolExecutor(max_workers=process_num) as pool:
        while True:
            try:
                if len(process_stack) < process_num:
                    add_functions = list(itertools.islice(functions, process_size - len(process_stack)))
                    if add_functions:
                        process_stack.extend(add_functions)
                    elif len(process_stack) == 0:
                        break
                task = process_stack.pop()
                futures.append(pool.submit(task))
            except concurrent.futures.process.BrokenProcessPool:
                break

    return [future.result() for future in futures if not future.exception()]


def blocking_pool(futures: List) -> Any:
    """
    This function - given a list of features - will wait for the result and return when all futures are resolved.

    :param: futures: This is a list of process futures.
    """
    to_return = []
    for future in futures:
        to_return.append(future.get())
    return to_return


def resolve_abs_file(path_to_file: str, file_name: str) -> str:
    """
    Resolve the absolute path of the file.

    :param path_to_file: The path to the file.
    :param file_name: The file name.
    :return: The absolute path of the file.
    """
    if not os.path.isabs(file_name):
        file_name = os.path.join(os.path.dirname(path_to_file), file_name)
    return file_name


def get_class(kls: str, kwargs) -> object:
    """
    This function provided a full class name including the module will return and instance
    of the class requested.

    :param: kls: Full class name, e.g. example encoder.encoder.IdentityEncoder.
    :param: kwargs: class constructor arguments.
    :return: A instance of specified class
    """
    parts = kls.split('.')
    start_i = 1 if parts[0] == 'censyn' else 0
    m = __import__('censyn')
    for comp in parts[start_i:]:
        m = getattr(m, comp)
    return m(**kwargs)


def find_all_subclasses(base_cls) -> List:
    """
    Returns all of the subclasses recursively for the given base class.

    :param base_cls: the base class to find all descendants of
    :return: A list of all subclasses
    """
    all_sub = []

    for subclass in base_cls.__subclasses__():
        all_sub.append(f'{subclass.__module__}.{subclass.__name__}')
        all_sub.extend(find_all_subclasses(subclass))
    return all_sub


def find_class(base_cls, target, kwargs) -> object:
    """
    Finds a class by name and returns an instantiated instance of it.

    Example Usage: get_class_based_on_class_name(Filter, 'ColumnFilter', {})

    :param base_cls: A base class from where to start looking.
    :param target: A class to look for, eg 'ColumnFilter'.
    :param kwargs: A dict of arguments to make a instance of the class.
    :return: An instantiated instance of the class.
    :raises: ValueError if this finds zero or more than one matching classes.
    """
    if not target:
        raise ValueError('Must provide a name to find')

    all_sub = find_all_subclasses(base_cls)
    if '.' not in target:
        target = '.' + target
    matching = [sub for sub in all_sub if target in sub]

    if len(matching) == 0:
        raise ValueError(f'Did not find class with the name {target}.')
    if len(matching) == 1:
        return get_class(matching[0], kwargs)
    raise ValueError('Provided ambiguous class string.')


def chunks(length: int, number_of_bins: int):
    """Yield successive size-sized chunks from length."""
    number_of_bins = math.ceil(len(length) / number_of_bins)
    for i in range(0, len(length), number_of_bins):
        x = length[i:i + number_of_bins]
        yield [x[0], x[-1]]


def frequent_item_set(individual_marginal_scores: List, support: float):
    """
    This is used to call the frequent item set library.

    :param:individual_marginal_scores: A list of items that to perform frequent item set analysis on.
    :param: support: The support threshold for the frequent item set analysis.
    :return: A DataFrame containing all the information from the frequent item set analysis.
    """
    encoder = TransactionEncoder()
    item_list = [item[0] for item in individual_marginal_scores]
    encoder_array = encoder.fit(item_list).transform(item_list)
    df = pd.DataFrame(encoder_array, columns=encoder.columns_)
    item_set_df = apriori(df, min_support=support, use_colnames=True)
    if item_set_df.empty:
        return pd.DataFrame(columns=['itemsets', 'count', 'sum score', 'average score'])
    item_set_df['count'] = pd.Series(item_set_df['support'] * len(item_list), dtype=int)
    item_set_df['itemsets'] = item_set_df['itemsets'].apply(list)
    item_set_df['itemsets'] = item_set_df['itemsets'].apply(sorted)
    item_set_df['sum score'] = pd.Series(data=0.0, dtype=float)
    for index, row in item_set_df.iterrows():
        mask = pd.Series(data=True, index=df.index)
        for feat in row['itemsets']:
            mask = mask & df[feat]
        feat_df = df[mask]
        item_set_df.at[index, 'sum score'] = sum([individual_marginal_scores[i][1] for i in feat_df.index])
    item_set_df['average score'] = item_set_df['sum score'] / item_set_df['count']
    item_set_df.drop('support', axis=1, inplace=True)
    return item_set_df


def remove_indicated_values(indicator_name: str, column_name : str, df_create_mask_from: pd.DataFrame,
                            df_to_remove_from: pd.DataFrame):
    """
    Remove the indicated value from the column_name in df_to_remove_from using a mask created using the
    indicator name and df_create_mask_from. This is then set to the value of 'None'.

    :param: indicator_name: The column of the indicator name.
    :param: column_name: The feature name to modify.
    :param: df_create_mask_from: The data frame to create the mask using the indicator name column.
    :param: df_to_remove_from: The data frame to modify.
    """
    if indicator_name not in df_create_mask_from.columns:
        raise ValueError(f'{indicator_name} not in DataFrame {df_create_mask_from.columns}')
    if column_name not in df_to_remove_from.columns:
        raise ValueError(f'{column_name} not in DataFrame {df_to_remove_from.columns}')
    assert df_create_mask_from.shape[0] > 0 and df_create_mask_from.shape[1] > 0,  \
        f'Error data must be not empty for {indicator_name}'
    df_to_remove_from.loc[(df_create_mask_from[indicator_name]), column_name] = None


def calculate_weights(in_df: pd.DataFrame, features: List[str]) -> pd.Series:
    """
    Calculate the weight using the data and the names of the weight features.

    :param: in_df: Data set with the weight features.
    :param: features: The weight features' names
    :return: Series of the weights
    """
    weight_data = None
    if len(features) > 0:
        weight_data = pd.Series(data=1, index=in_df.index)
        for feat in features:
            weight_data = weight_data.copy().mul(in_df[feat])
        weight_data.name = features[0]
    return weight_data


class NotValidator:
    def __init__(self, validator) -> None:
        self.validator = validator

    def __call__(self, value) -> bool:
        return not self.validator(value)


class ValidatorValueType:
    def __init__(self, enum_types) -> None:
        self.enum_types = enum_types

    def __call__(self, value) -> bool:
        return isinstance(value, self.enum_types)


class FalseValidator:
    def __call__(self) -> bool:
        return False
