#pd.Dataframe

def make_df_from_dir(dir, columns=['filename']):
  data = {}
  for column in columns:
    data[column] = []

  for item in os.listdir(dir):
    data[columns[0]].append(f'{dir}/{item}')

    # with open(f'{DATA_DIR}/Labels/{item[:-5]}.txt', 'r') as f:
    #       file_content = f.read()

  return pd.DataFrame(data)