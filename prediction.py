import pickle
from pathlib import Path
import xgboost as xgb
from typing import Dict, Any


_DIR = Path(__file__).parent
MODEL_PATH = _DIR / "model.ubj"
VECTORIZER_PATH = _DIR / "vectorizer.pkl"


class Predictor:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))
        with open(VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text: str) -> Dict[str, Any]:
        features = self.vectorizer.transform([text]).toarray()
        d_matrix = xgb.DMatrix(features)
        probability = self.model.predict(d_matrix)[0]

        return {
            "prediction": 'SPAM' if probability >= 0.5 else 'NOT SPAM',
            "probability": probability,
            "confidence": (
                probability
                if probability >= 0.5
                else 1.0 - probability
            )
        }
