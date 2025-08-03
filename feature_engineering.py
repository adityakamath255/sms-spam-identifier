import re
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer


class FeatureExtractor:
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

    def __init__(self, lemmatizer: WordNetLemmatizer):
        self.lemmatizer = lemmatizer

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by lowering case, removing stopwords,
        and lemmatizing."""
        text = text.lower()
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract additional features such as URL count,
        special character ratio, etc."""
        lower_text = text.lower()
        num_words = max(1, len(text.split()))  # avoid division by zero
        text_length = max(1, len(text))  # avoid division by zero

        urls = re.findall(self.URL_PATTERN, lower_text, re.VERBOSE)
        email_addrs = re.findall(self.EMAIL_PATTERN, lower_text, re.VERBOSE)
        phone_nos = re.findall(self.PHONE_PATTERN, lower_text, re.VERBOSE)
        capitals = [c for c in text if c.isupper()]
        digits = [c for c in text if c.isdigit()]
        specials = [c for c in text if c in self.SPECIAL_CHARS]

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

    def process_single(
        self,
        text: str,
        vectorizer: TfidfVectorizer
    ) -> np.ndarray:

        preprocessed_text = self.preprocess_text(text)
        tfidf_features = vectorizer.transform([preprocessed_text])
        addl_features = pd.DataFrame([self.extract_features(text)])
        all_features = np.hstack([
            tfidf_features.toarray(),
            addl_features.values
        ])

        return all_features

    def process_batch(
        self,
        texts,
        vectorizer: TfidfVectorizer
    ) -> np.ndarray:

        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_features = vectorizer.fit_transform(preprocessed_texts)
        addl_features = pd.DataFrame(
            self.extract_features(text) for text in texts
        )
        all_features = np.hstack([
            tfidf_features.toarray(),
            addl_features.values
        ])
        return all_features
