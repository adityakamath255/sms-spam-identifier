import re
import numpy as np
import pandas as pd
from typing import Dict, Any


URL_PATTERN = r"""
    (https?:\/\/)?               # optional http:// or https://
    (?:www\.)?                   # optional www.
    [a-zA-Z0-9-]+                # domain name
    \.                           # dot before the TLD
    [a-zA-Z0-9-]{2,}             # top-level domain (TLD)
    (\/[^\s]*)?                  # optional path after the domain
"""

EMAIL_PATTERN = r"""
    [a-zA-Z0-9._%+-]+            # local part of the email (before @)
    @                            # the @ symbol
    [a-zA-Z0-9.-]+               # domain name
    \.                           # dot before the TLD
    [a-zA-Z0-9.-]{2,}            # top-level domain (TLD)
"""

PHONE_PATTERN = r'\d{7,}'

SPECIAL_CHARS = '@#$%^&*~`{}[]<>|\\'


def _preprocess_text(text: str, lemmatizer: Any) -> str:
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def _extract_features(text: str) -> Dict[str, float]:
    num_words = max(1, len(text.split()))  # avoid division by zero
    text_length = max(1, len(text))  # avoid division by zero

    urls = re.findall(URL_PATTERN, text, re.VERBOSE)
    email_addrs = re.findall(EMAIL_PATTERN, text, re.VERBOSE)
    phone_nos = re.findall(PHONE_PATTERN, text, re.VERBOSE)
    capitals = [c for c in text if c.isupper()]
    digits = [c for c in text if c.isdigit()]
    specials = [c for c in text if c in SPECIAL_CHARS]

    # dividing by text_length for single character level features
    # dividing by num_words for multiple character level features
    return {
        'length': text_length,
        'num_words': num_words,
        'unique_word_count': len(set(text.split())),
        'url_ratio': len(urls) / num_words,
        'email_ratio': len(email_addrs) / num_words,
        'phone_ratio': len(phone_nos) / num_words,
        'uppercase_ratio': len(capitals) / text_length,
        'digits_ratio': len(digits) / text_length,
        'special_ratio': len(specials) / text_length,
        'exclamation_ratio': text.count('!') / text_length,
    }


def extract_features_single(
    text: str,
    vectorizer: Any,
    lemmatizer: Any,
) -> np.ndarray:
    preprocessed_text = _preprocess_text(text, lemmatizer)
    tfidf_features = vectorizer.transform([preprocessed_text])
    addl_features = pd.DataFrame([_extract_features(text)])
    all_features = np.hstack([
        tfidf_features.toarray(),
        addl_features.values
    ])

    return all_features


def extract_features_batch(
    texts,
    vectorizer: Any,
    lemmatizer: Any
) -> np.ndarray:
    preprocessed_texts = [_preprocess_text(text, lemmatizer) for text in texts]
    tfidf_features = vectorizer.fit_transform(preprocessed_texts)
    addl_features = pd.DataFrame(
        _extract_features(text) for text in texts
    )
    all_features = np.hstack([
        tfidf_features.toarray(),
        addl_features.values
    ])
    return all_features
