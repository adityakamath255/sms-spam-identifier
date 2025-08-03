import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from nltk.stem import WordNetLemmatizer

from feature_engineering import FeatureExtractor


class ModelTrainer:
    format_spec = {
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

    def __init__(self, label_map: Dict[str, int] = {"ham": 0, "spam": 1}):
        self.label_map = label_map
        self.feature_extractor = FeatureExtractor(WordNetLemmatizer())
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 4),
            max_df=0.70,
            min_df=2,
            use_idf=True
        )

    def load_data(self, filepath: str) -> Tuple[pd.Series, pd.Series]:
        df = pd.read_csv(filepath, encoding="latin1")
        messages = df["TEXT"]
        messages.fillna("", inplace=True)
        labels = df["LABEL"].map(self.label_map)
        return messages, labels

    def prepare_dataset(self, messages: pd.Series, labels: pd.Series) -> tuple:
        features = self.feature_extractor.process_batch(
            messages, self.vectorizer
        )
        return train_test_split(
            features,
            labels,
            test_size=0.2,
            stratify=labels
        )

    def train_model(self, X: np.ndarray, y: np.ndarray) -> xgb.Booster:
        """Train the model."""
        dtrain = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(X, label=y)

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": np.sum(y == 0) / np.sum(y == 1)
        }

        evals = [(dtrain, 'train'), (dval, 'validation')]

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            verbose_eval=50,
            early_stopping_rounds=100
        )

        return model

    def evaluate_model(
        self,
        model: xgb.Booster,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""

        dtest = xgb.DMatrix(X)
        pred_probs = model.predict(dtest)
        pred_labels = (pred_probs > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred_labels, average='binary'
        )
        accuracy = accuracy_score(y, pred_labels)

        # evaluate lift
        sorted_indices = np.argsort(pred_probs)[::-1]
        sorted_y_test = y.iloc[sorted_indices]
        top_percentage = 0.2
        top_n = int(len(sorted_y_test) * top_percentage)
        baseline_rate = np.mean(y)
        lift = (np.mean(sorted_y_test[:top_n]) / baseline_rate)

        tn, fp, fn, tp = confusion_matrix(y, pred_labels).ravel()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'lift': lift,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }

    def print_metrics(self, metrics: Dict[str, float]):
        for metric, format_spec in self.format_spec.items():
            value = metrics[metric]
            print(f"{metric}: {value:{format_spec}}")

    def save_artifacts(
        self,
        model: xgb.Booster,
        model_path: str,
        vectorizer_path: str
    ):
        model.save_model(model_path)
        with open(vectorizer_path, 'wb') as file:
            pickle.dump(self.vectorizer, file)

    def run_training_pipeline(
        self,
        data_path: str = "spam.csv",
        model_path: str = "model.ubj",
        vectorizer_path: str = "vectorizer.ubj"
    ) -> Tuple[xgb.Booster, Dict[str, float]]:

        messages, labels = self.load_data(data_path)
        dataset = self.prepare_dataset(messages, labels)
        X_train, X_test, y_train, y_test = dataset
        model = self.train_model(X_train, y_train)
        self.save_artifacts(model, model_path, vectorizer_path)
        metrics = self.evaluate_model(model, X_test, y_test)
        return (model, metrics)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
