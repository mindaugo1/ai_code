import attr


class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def create_data_bundle_obj_from_dataframe(cls, dataframe, x_col, y_col):
        return cls(dataframe[x_col].values, dataframe[y_col].values)

    @classmethod
    def split(cls, data_bundle, fracs, random=True):
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
            split_data_bundle = (
                x[current_index: current_index + dx],
                y[current_index: current_index + dx],
            )
            result.append(split_data_bundle)
            current_index += dx

        return tuple(result)


@attr.s(auto_attribs=True)
class DataContainer:
    train: DataBundle
    validation: DataBundle
    test: DataBundle


def make_dataset(data_bundle):
    return tf.data.Dataset.from_tensor_slices((data_bundle.x, data_bundle.y))


data_container.train.dataset = make_dataset(data_container.train)
data_container.validation.dataset = make_dataset(data_container.validation)
data_container.test.dataset = make_dataset(data_container.test)
