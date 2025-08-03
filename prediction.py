import pickle
import xgboost as xgb
from typing import Dict
from nltk.stem import WordNetLemmatizer

from feature_engineering import FeatureExtractor


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
        self.feature_extractor = FeatureExtractor(WordNetLemmatizer())

    def predict(self, text: str) -> Dict[str, float]:
        """Predict whether the message is spam or not."""
        features = self.feature_extractor(text, self.vectorizer)

        probability = self.model.predict(xgb.DMatrix(features))[0]

        return {
            "prediction": 'SPAM' if probability >= 0.5 else 'NOT SPAM',
            "probability": probability,
            "confidence": (
                probability
                if probability >= 0.5
                else 1.0 - probability
            )
        }
