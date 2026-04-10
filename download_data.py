import pandas as pd
import nltk
import zipfile
import os
from urllib.request import urlretrieve

DB_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def download_nltk_data():
    nltk.download('wordnet')


def download_dataset():
    urlretrieve(DB_URL, "dataset.zip")

    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extract("SMSSpamCollection")

    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None)
    df.columns = ["LABEL", "TEXT"]
    df.to_csv("spam.csv", index=False)

    os.remove("SMSSpamCollection")
    os.remove("dataset.zip")


if __name__ == "__main__":
    download_nltk_data()
    download_dataset()
