
def drop_values(df, col_name, values):
    return df.loc[~df[col_name].isin(values), :].reset_index(drop=True)


def drop_rare_values(df, col_name, threshold):
    counts = df[col_name].value_counts(normalize=True)
    return df.loc[df[col_name].isin(counts[counts > threshold].index), :].reset_index(
        drop=True
    )