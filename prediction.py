import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple
from nltk.stem import WordNetLemmatizer

from model_trainer import *


class Predictor:
    def __init__(
        self,
        model_path: str = "model.ubj",
        vectorizer_path: str = "vectorizer.ubj"
    ):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.lemmatizer = WordNetLemmatizer()

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict whether the message is spam or not."""
        processed_text = preprocess_text(text, self.lemmatizer)
        features = self.vectorizer.transform([processed_text])
        additional_features = pd.DataFrame([extract_features(text)])
        combined_features = np.hstack([
            features.toarray(),
            additional_features.values
        ])

        probability = self.model.predict(xgb.DMatrix(combined_features))[0]

        return {
            "prediction": 'SPAM' if probability >= 0.5 else 'NOT SPAM',
            "probability": probability,
            "confidence": (
                probability
                if probability >= 0.5
                else 1.0 - probability
            )
        }
