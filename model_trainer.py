import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
from typing import Tuple, Dict
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")

URL_PATTERN = r"""
  (https?:\/\/)?               # optional http:// or https://
  (?:www\.)?                   # optional www.
  [a-zA-Z0-9-]+                # domain name (alphanumeric characters and hyphens)
  \.                           # dot before the TLD
  [a-zA-Z0-9-]{2,}             # top-level domain (TLD) like .com, .edu, etc.
  (\/[^\s]*)?                  # optional path after the domain (e.g., /path)                          
"""

EMAIL_PATTERN = r"""
  [a-zA-Z0-9._%+-]+            # local part of the email (before @)
  @                            # the @ symbol
  [a-zA-Z0-9.-]+               # domain name part (alphanumeric, dots, and hyphens)
  \.                           # dot before the TLD
  [a-zA-Z0-9.-]{2,}            # top-level domain (TLD) like .com, .org, etc.
"""

PHONE_PATTERN = r'\d{7,}'

SPECIAL_CHARS = '@#$%^&*~`{}[]<>|\\'

def load_data(filepath: str, label_map: Dict[str, int]) -> Tuple[pd.Series, pd.Series]:
    """Load data from a CSV file and map labels."""
    try:
        df = pd.read_csv(filepath, encoding='latin1')
        messages = df['TEXT']
        messages.fillna('', inplace=True) 
        labels = df['LABEL'].map(label_map)
        print(f"Loaded {len(df)} messages from {filepath}")
        return messages, labels
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """Preprocess text by lowering case, removing stopwords, and lemmatizing."""
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def extract_features(text: str) -> dict:
    """Extract additional features such as URL count, special character ratio, etc."""
    lower_text = text.lower()
    num_words = max(1, len(text.split())) # avoid division by zero
    text_length = max(1, len(text)) # avoid division by zero

    # dividing by text_length for single character level features
    # dividing by num_words for multiple character level features
    return {
        'length': text_length,
        'num_words': num_words,
        'unique_word_count': len(set(text.split())),
        'url_ratio': len(re.findall(URL_PATTERN, lower_text, re.VERBOSE)) / num_words,
        'email_ratio': len(re.findall(EMAIL_PATTERN, lower_text, re.VERBOSE)) / num_words,
        'phone_ratio': len(re.findall(PHONE_PATTERN, text)) / num_words,
        'uppercase_ratio': sum(c.isupper() for c in text) / text_length,
        'digits_ratio': sum(c.isdigit() for c in text) / text_length,
        'special_ratio': sum(text.count(c) for c in SPECIAL_CHARS) / text_length,
        'exclamation_ratio': text.count('!') / text_length,
    }

def prepare_dataset(messages: pd.Series, labels: pd.Series, vectorizer: TfidfVectorizer, 
                   lemmatizer: WordNetLemmatizer) -> tuple:
    """Prepare the dataset by preprocessing text and extracting features."""
    processed_texts = messages.apply(lambda x: preprocess_text(x, lemmatizer))
    tfidf_features = vectorizer.fit_transform(processed_texts)
    additional_features = pd.DataFrame(messages.apply(extract_features).tolist())
    features = np.hstack([tfidf_features.toarray(), additional_features.values])
    return train_test_split(features, labels, test_size=0.2, stratify=labels)

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.Booster:
    """Train the model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_train, label=y_train)  # can use the same dataset as validation for this example
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 5,
        "eta": 0.05,  # learning rate
        "subsample": 0.85,  # subsampling ratio
        "colsample_bytree": 0.85,  # column subsampling ratio
        "scale_pos_weight": np.sum(y_train == 0) / np.sum(y_train == 1)  # class imbalance handling
    }
    
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000, # number of boosting rounds
        evals=evals, # validation set for evaluation metrics
        verbose_eval=50, # display progress every 50 rounds
        early_stopping_rounds=100 # stop early if no improvement
    )
    
    return model

def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance."""
    dtest = xgb.DMatrix(X_test)
    pred_probs = model.predict(dtest)
    pred_labels = (pred_probs > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_labels, average='binary'
    )
    accuracy = accuracy_score(y_test, pred_labels)

    # evaluate lift
    sorted_indices = np.argsort(pred_probs)[::-1]
    sorted_y_test = y_test.iloc[sorted_indices]
    top_percentage = 0.2
    top_n = int(len(sorted_y_test) * top_percentage)
    baseline_rate = np.mean(y_test)
    lift = (np.mean(sorted_y_test[:top_n]) / baseline_rate)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_labels).ravel()

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

def print_metrics(metrics: dict) -> None:
    """Print the evaluation metrics in a readable format."""
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

    for metric, format_spec in format_spec.items():
        value = metrics[metric]
        print(f"{metric}: {value:{format_spec}}")

def save_model(model: xgb.Booster, vectorizer: TfidfVectorizer, 
               model_path: str, vectorizer_path: str) -> None:
    """Save the trained model and vectorizer to disk."""
    model.save_model(model_path)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Model saved to {model_path} and vectorizer to {vectorizer_path}")

def main():
    """Main function to run the spam classification pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 4),
        max_df=0.70,
        min_df=2,
        use_idf=True
    )
    lemmatizer = WordNetLemmatizer()

    try:
        messages, labels = load_data("./spam.csv", {"ham": 0, "spam": 1})

        X_train, X_test, y_train, y_test = prepare_dataset(
            messages, labels, vectorizer, lemmatizer
        )

        print("Training model...")
        model = train_model(X_train, y_train)
        save_model(model, vectorizer, "model.ubj", "vectorizer.ubj")
    
        print("\n---\n")

        metrics = evaluate_model(model, X_test, y_test)
        print_metrics(metrics)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
