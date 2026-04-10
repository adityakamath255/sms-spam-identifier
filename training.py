import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from nltk.stem import WordNetLemmatizer
from pprint import pprint

from feature_engineering import extract_features_batch

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 4)
MAX_DF = 0.70
MIN_DF = 2
NUM_BOOST_ROUNDS = 1000
VERBOSE_EVAL = 50
EARLY_STOPPING_ROUNDS = 100
TOP_PERCENTAGE = 0.2

MAX_DEPTH = 5
LEARNING_RATE = 0.05
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85

TEST_SIZE = 0.25
RANDOM_STATE = 100

LABEL_MAP = {
    "ham": 0,
    "spam": 1
}

_DIR = Path(__file__).parent
FILEPATH = _DIR / "spam.csv"
MODEL_PATH = _DIR / "model.ubj"
VECTORIZER_PATH = _DIR / "vectorizer.ubj"

FORMAT_SPEC = {
    "accuracy": ".2%",
    "precision": ".2%",
    "recall": ".2%",
    "f1_score": ".2%",
    "lift": ".4f",
    "true_negatives": "d",
    "false_positives": "d",
    "false_negatives": "d",
    "true_positives": "d"
}


def get_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        max_df=MAX_DF,
        min_df=MIN_DF,
        use_idf=True
    )


def load_data() -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(FILEPATH, encoding="latin1")
    messages = df["TEXT"]
    messages.fillna("", inplace=True)
    labels = df["LABEL"].map(LABEL_MAP)
    return messages, labels


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": MAX_DEPTH,
        "eta": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "scale_pos_weight": np.sum(y_train == 0) / np.sum(y_train == 1)
    }

    evals = [(dtrain, 'train'), (dval, 'validation')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        verbose_eval=VERBOSE_EVAL,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    return model


def evaluate_model(
    model: xgb.Booster,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    dtest = xgb.DMatrix(X)
    pred_probs = model.predict(dtest)
    pred_labels = (pred_probs > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred_labels, average='binary'
    )
    accuracy = accuracy_score(y, pred_labels)

    tn, fp, fn, tp = confusion_matrix(y, pred_labels).ravel()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }


def print_metrics(metrics: Dict[str, float]):
    for metric, format_spec in FORMAT_SPEC.items():
        value = metrics[metric]
        print(f"{metric}: {value:{format_spec}}")


def save_artifacts(
    model: xgb.Booster,
    vectorizer: Any,
):
    model.save_model(MODEL_PATH)
    with open(VECTORIZER_PATH, 'wb') as file:
        pickle.dump(vectorizer, file)


def run_training_pipeline() -> Tuple[xgb.Booster, Dict[str, float]]:
    messages, labels = load_data()
    lemmatizer = WordNetLemmatizer()
    vectorizer = get_vectorizer()
    msg_train, msg_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    X_train = extract_features_batch(msg_train, vectorizer, lemmatizer, fit=True)
    X_test = extract_features_batch(msg_test, vectorizer, lemmatizer, fit=False)
    model = train_model(X_train, y_train, X_test, y_test)
    save_artifacts(model, vectorizer)
    metrics = evaluate_model(model, X_test, y_test)
    return (model, metrics)


if __name__ == "__main__":
    _, metrics = run_training_pipeline()
    print(f"Model saved at {MODEL_PATH}")
    print("Model specs:")
    pprint(metrics)
