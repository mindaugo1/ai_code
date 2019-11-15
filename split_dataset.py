class DataBundle:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @classmethod
    def from_dataframe(cls, dataframe, x_col, y_col):
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
