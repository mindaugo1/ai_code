import attr
import tensorflow as tf
import numpy as np
import pandas as pd
import math
tf.enable_eager_execution()


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def create_data_bundle_from_unbalanced_data_bundle(cls, data_bundle):
        values, counts = np.unique(data_bundle.y, return_counts=True)
        max_count = counts.max()
        x = []
        y = []
        for value, count in zip(values, counts):
            indices = np.argwhere(data_bundle.y == value)
            for _ in range(max_count // count):
                x.append(data_bundle.x[indices].flatten())
                y.append(data_bundle.y[indices].flatten())

        return cls(np.concatenate(x), np.concatenate(y))

    @classmethod
    def create_data_bundle_obj_from_dataframe(cls, dataframe, x_col, y_col):
        return cls(dataframe[x_col].values, dataframe[y_col].values)

    @classmethod
    def split_data_bundle_obj(cls, data_bundle, fracs, random=True):
        x = data_bundle.x
        y = data_bundle.y

        if random:
            random_indices = np.random.permutation(len(y))
            x = x[random_indices]
            y = y[random_indices]

        result = []
        current_index = 0

        for frac in fracs:
            dx = math.ceil(len(y) * frac)
            temp_data_bundle_objs_container = cls(
                x=x[current_index: current_index + dx],
                y=y[current_index: current_index + dx],
            )

            result.append(temp_data_bundle_objs_container)
            current_index += dx

        return tuple(result)


@attr.s(auto_attribs=True)
class DataContainer:
    train: DataBundle
    validation: DataBundle
    test: DataBundle
