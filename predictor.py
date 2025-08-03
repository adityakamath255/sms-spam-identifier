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
        pred_prob = self.model.predict(xgb.DMatrix(combined_features))[0]
        return 'SPAM' if pred_prob > 0.5 else 'NOT SPAM', pred_prob


def cli(pred: Predictor) -> None:
    """Interactively classify user-inputted messages as spam or not."""
    try:
        while True:
            message = input("\nEnter message to classify (or 'quit'): ")
            if message.lower() == 'quit':
                break

            prediction, probability = pred.predict(message)
            confidence = probability if probability > 0.5 else 1 - probability
            print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

    except KeyboardInterrupt:
        print("\nProgram terminated by user")


def main():
    """Main entry point for running the spam detection program."""
    try:
        pred = Predictor()
        cli(pred)
    except Exception as e:
        print(f"Error: {str(e)}")
