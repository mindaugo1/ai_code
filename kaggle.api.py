DOWNLOAD_DATA = True
if DOWNLOAD_DATA:
    !rm - rf data
    !rm - fr temp
    from google.colab import drive
    drive.mount('/content/drive')
    !mkdir .kaggle/
    !cp ./drive/My\ Drive/'Colab Notebooks'/kaggle/kaggle.json / root/.kaggle/
    !chmod 600 / root/.kaggle/kaggle.json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('jrobischon/wikipedia-movie-plots', path=DATA_DIR, unzip=True)
