# pd.Dataframe


def make_df_from_dir(dir, columns=['filename']):
    data = {}
    for column in columns:
        data[column] = []

    for item in os.listdir(dir):
        data[columns[0]].append(f'{dir}/{item}')

        # with open(f'{DATA_DIR}/Labels/{item[:-5]}.txt', 'r') as f:
        #       file_content = f.read()

    return pd.DataFrame(data)

    def get_image_dims(arr):
        heights, widths = [], []
        for filename in arr:
            height, width, _ = np.array(keras.preprocessing.image.load_img(filename)).shape
            heights.append(height)
            widths.append(width)
        return np.asarray(heights), np.asarray(widths)
