from typing import Dict

import pandas as pd


def set_data_type(in_df: pd.DataFrame, features: Dict) -> pd.DataFrame:
    """
    Set the DataFrame data to match the feature_type value.
    Feature type of integer are stored as float to enable NA values.

    :param in_df: DataFrame to validate.
    :param features: The dictionary of features
    :return: The validated DataFrame
    """
    for f_name, feature in features.items():
        if f_name in in_df.columns:
            in_df[f_name] = feature.set_series_data_type(in_s=in_df[f_name])
            if feature.feature_format:
                in_df[f_name] = feature.calculate_feature_data(data_df=in_df, expr=feature.feature_format)
    return in_df
