import attr


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

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
            temp_data_bundle_obj_container = cls(
                x=x[current_index: current_index + dx],
                y=y[current_index: current_index + dx],
            )

            result.append(temp_data_bundle_obj_container)
            current_index += dx

        return tuple(result)

    def make_tf_dataset_from_data_bundle(data_bundle):
        return tf.data.Dataset.from_tensor_slices((data_bundle.x, data_bundle.y))


@attr.s(auto_attribs=True)
class DataContainer:
    train: DataBundle
    validation: DataBundle
    test: DataBundle
