import re
import numpy as np
import pandas as pd
from typing import Any, Dict


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


def _extract_features(text: str) -> Dict[str, float]:
    words = text.split()
    num_words = max(1, len(words))
    text_length = max(1, len(text))

    return {
        'length': text_length,
        'num_words': num_words,
        'unique_word_count': len(set(words)),
        'url_ratio': len(re.findall(URL_PATTERN, text, re.VERBOSE)) / num_words,
        'email_ratio': len(re.findall(EMAIL_PATTERN, text, re.VERBOSE)) / num_words,
        'phone_ratio': len(re.findall(PHONE_PATTERN, text)) / num_words,
        'uppercase_ratio': sum(c.isupper() for c in text) / text_length,
        'digits_ratio': sum(c.isdigit() for c in text) / text_length,
        'special_ratio': sum(c in SPECIAL_CHARS for c in text) / text_length,
        'exclamation_ratio': text.count('!') / text_length,
    }


def extract_features_single(
    text: str,
    vectorizer: Any,
) -> np.ndarray:
    tfidf_features = vectorizer.transform([text])
    addl_features = np.array([list(_extract_features(text).values())])
    return np.hstack([tfidf_features.toarray(), addl_features])


def extract_features_batch(
    texts,
    vectorizer: Any,
    fit: bool = True,
) -> np.ndarray:
    if fit:
        tfidf_features = vectorizer.fit_transform(texts)
    else:
        tfidf_features = vectorizer.transform(texts)
    addl_features = pd.DataFrame(
        _extract_features(text) for text in texts
    )
    all_features = np.hstack([
        tfidf_features.toarray(),
        addl_features.values
    ])
    return all_features
