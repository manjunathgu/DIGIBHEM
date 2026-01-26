"""
Fake News Detection Project

This script trains a machine learning model to classify news as fake or real.

How to Run:
1. Ensure the dataset `fake_news.csv` is in the same folder.
2. Run the script: `python fake_news.py`
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# File paths
DATASET_PATH = "dataset/fake_news.csv"
MODEL_PATH = "fake_news_model.pkl"

def load_dataset():
    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")
    fake['label'] = 'fake'
    true['label'] = 'real'
    return pd.concat([fake, true], ignore_index=True)


def preprocess_text(text):
    """Clean and preprocess text data."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove unwanted characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def extract_features(texts):
    """Convert text data into numerical features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts), vectorizer

def train_models(X_train, y_train):
    """Train Logistic Regression and Naive Bayes models."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB()
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, path):
    """Save the trained model to a file."""
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def main():
    # Load dataset
    data = load_dataset()
    # Preprocess text data
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['label'], test_size=0.2, random_state=42
    )
    # Extract features using TF-IDF
    X_train_tfidf, vectorizer = extract_features(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    # Train models
    models = train_models(X_train_tfidf, y_train)
    # Evaluate models and select the best one
    best_model, best_f1 = None, 0
    for name, model in models.items():
        print(f"\n{name}:")
        evaluate_model(model, X_test_tfidf, y_test)
        f1 = classification_report(y_test, model.predict(X_test_tfidf), output_dict=True)['weighted avg']['f1-score']
        if f1 > best_f1:
            best_f1, best_model = f1, model
    # Save the best model
    save_model(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    # Example predictions
    new_texts = ["This is a completely fake news article.", "This is a genuine and real news story."]
    processed_texts = [preprocess_text(text) for text in new_texts]
    predictions = best_model.predict(vectorizer.transform(processed_texts))
    for text, label in zip(new_texts, predictions):
        print(f"Text: {text}\nPredicted Label: {label}\n")

if __name__ == "__main__":
    main()