class TransformingDataframes:

    import numpy as np

    def extract_columns_values_to_one_column(dataframe, columns_keys, content_col, features_col):
        result = []
        for column in columns_keys.keys():
            value = columns_keys.get(column)
            temp_dataframe = pd.DataFrame(columns=[features_col, content_col])
            temp_dataframe[features_col] = dataframe[column].values
            temp_dataframe[content_col] = value
            result.append(temp_dataframe)
        pd_result = pd.concat(result)
        return pd_result.loc[pd_result[features_col].notnull()].reset_index(drop=True)

    def count_not_null_values(dataframe, columns):
        not_null_values = 0
        for col in columns:
            not_null = dataframe[col].notnull().sum()
            not_null_values += not_null
        print(not_null_values)

    def drop_values_from_dataframe(dataframe, col_name, values_to_drop):
        return dataframe.loc[~dataframe[col_name].isin(values_to_drop), :].reset_index(drop=True)

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
