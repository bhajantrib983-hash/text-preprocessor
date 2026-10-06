# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:24:39 2026

@author: Balappa
"""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources once, if needed:
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)


def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Tokenize the text
    tokens = word_tokenize(text)

    # 3. Keep words only, excluding punctuation and stopwords
    cleaned_tokens = [
        word for word in tokens
        if word not in STOP_WORDS
        and word not in PUNCTUATION
        and any(char.isalnum() for char in word)
    ]

    stats = {
        "original_tokens": len(tokens),
        "cleaned_words": len(cleaned_tokens),
        "removed_tokens": len(tokens) - len(cleaned_tokens)
    }

    return cleaned_tokens, stats


if __name__ == "__main__":
    text = input("Enter text: ")

    cleaned_words, stats = preprocess_text(text)

    print("\nOriginal tokens:", stats["original_tokens"])
    print("Cleaned words:", stats["cleaned_words"])
    print("Removed tokens:", stats["removed_tokens"])
    print("Cleaned text:", cleaned_words)
    