
import numpy as np


def drop_values_from_dataframe(dataframe, col_name, values_to_keep):
    return dataframe.loc[~dataframe[col_name].isin(values_to_keep), :].reset_index(drop=True)


def drop_rare_values_from_dataframe(dataframe, col_name, threshold):
    counts = dataframe[col_name].value_counts(normalize=True)
    return dataframe.loc[dataframe[col_name].isin(counts[counts > threshold].index), :].reset_index(drop=True)


class_weights = dict(
    enumerate(
        sk.utils.class_weight.compute_class_weight(
            "balanced", np.unique(data_container.train.y), data_container.train.y
        )
    )
)
