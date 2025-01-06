import pickle
import xgboost as xgb
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from model_trainer import *

def load_saved_model(model_path: str, vectorizer_path: str) -> Tuple[xgb.Booster, TfidfVectorizer]:
    """Load a previously saved model and vectorizer from disk."""
    model = xgb.Booster()
    model.load_model(model_path)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict(text: str, model: xgb.Booster, 
            vectorizer: TfidfVectorizer, lemmatizer: WordNetLemmatizer) -> Tuple[str, float]:
    """Predict whether the message is spam or not."""
    processed_text = preprocess_text(text, lemmatizer)
    features = vectorizer.transform([processed_text])
    additional_features = pd.DataFrame([extract_features(text)])
    combined_features = np.hstack([features.toarray(), additional_features.values])
    
    pred_prob = model.predict(xgb.DMatrix(combined_features))[0]
    return 'SPAM' if pred_prob > 0.5 else 'NOT SPAM', pred_prob

def prediction_loop(model: xgb.Booster, vectorizer: TfidfVectorizer, lemmatizer: WordNetLemmatizer) -> None:
    """Interactively classify user-inputted messages as spam or not."""
    try:
        while True:
            message = input("\nEnter message to classify (or 'quit'): ")
            if message.lower() == 'quit':
                break
                
            prediction, probability = predict(message, model, vectorizer, lemmatizer)
            confidence = probability if probability > 0.5 else 1 - probability
            print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

def main():
    """Main entry point for running the spam detection program."""
    try:
        model, vectorizer = load_saved_model("model.ubj", "vectorizer.ubj")
        lemmatizer = WordNetLemmatizer()
        prediction_loop(model, vectorizer, lemmatizer)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
