class DataFrameUtils:

    import numpy as np
    import pandas as pd

    @staticmethod
    def create_new_x_y_dataframe_from_dataframe(dataframe, columns, labels, new_x_col_name, new_y_col_name):
        result = []
        for column, label in zip(columns, labels):
            temp_dataframe = pd.DataFrame(columns=[new_x_col_name, new_y_col_name])
            temp_dataframe[new_y_col_name] = dataframe[column].values
            temp_dataframe[new_x_col_name] = label
            result.append(temp_dataframe)

        pd_result = pd.concat(result)
        return pd_result.loc[pd_result[new_y_col_name].notnull()].reset_index(drop=True)

    @staticmethod
    def count_not_null_values(dataframe, columns):
        not_null_values = 0
        for col in columns:
            not_null = dataframe[col].notnull().sum()
            not_null_values += not_null
        print(not_null_values)

    @staticmethod
    def drop_values_from_dataframe(dataframe, col_name, values_to_drop):
        return dataframe.loc[~dataframe[col_name].isin(values_to_drop), :].reset_index(drop=True)

    @staticmethod
    def drop_rare_values_from_dataframe(dataframe, col_name, threshold):
        counts = dataframe[col_name].value_counts(normalize=True)
        return dataframe.loc[dataframe[col_name].isin(counts[counts > threshold].index), :].reset_index(drop=True)

    @staticmethod
    def make_category_map(labels):
        return {x: i for i, x in enumerate(set(labels))}

    @staticmethod
    def calculate_class_weights(data_container):
        return dict(
            enumerate(
                sk.utils.class_weight.compute_class_weight(
                    "balanced", np.unique(data_container.train.y), data_container.train.y
                )
            )
        )
