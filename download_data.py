import pandas as pd
import zipfile
import os
from pathlib import Path
from urllib.request import urlretrieve

DB_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

_DIR = Path(__file__).parent


def download_dataset():
    zip_path = _DIR / "dataset.zip"
    raw_path = _DIR / "SMSSpamCollection"
    csv_path = _DIR / "spam.csv"

    urlretrieve(DB_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("SMSSpamCollection", _DIR)

    df = pd.read_csv(raw_path, sep='\t', header=None)
    df.columns = ["LABEL", "TEXT"]
    df.to_csv(csv_path, index=False)

    os.remove(raw_path)
    os.remove(zip_path)


if __name__ == "__main__":
    download_dataset()
