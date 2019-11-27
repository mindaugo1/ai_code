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
    @classmethod
    def create_data_bundle_from_unbalanced_data_bundle(
        cls,
        data_bundle,
        target_class_size: int,
        value_counts: Dict[int, int],
    ):
        x = []
        y = []
        for label, current_size in value_counts.items():
            current_indices = np.argwhere(data_bundle.y == label).flatten()
            next_indices = []
            for _ in range(target_class_size // current_size):
                next_indices.append(np.random.permutation(current_indices))
            next_indices.append(
                np.random.choice(current_indices, target_class_size % current_size)
            )
            next_indices = np.concatenate(next_indices)
            x.append(data_bundle.x[next_indices])
            y.append(data_bundle.y[next_indices])

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
