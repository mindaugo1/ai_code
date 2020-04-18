# DOWNLOAD_DATA = True
# if DOWNLOAD_DATA:
#     shutil.rmtree(str(DATA_DIR))
#     shutil.rmtree(str(TEMP_DIR))
#     from google.colab import drive
#     drive.mount('/content/drive')
#     !mkdir .kaggle/
#     !cp /content/drive/My\ Drive/Colab\ Notebooks/kaggle/kaggle.json /content/.kaggle
#     !chmod 600 /content/.kaggle/kaggle.json
#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     TEMP_DIR.mkdir(parents=True, exist_ok=True)
#     import kaggle
#     kaggle.api.authenticate()
#     kaggle.api.dataset_download_files('jrobischon/wikipedia-movie-plots', path=DATA_DIR, unzip=True)
