import os
import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------------
# Load Dataset
# ------------------------
def load_data(folder_path):
    texts = []
    labels = []

    for label in ['pos', 'neg']:
        path = os.path.join(folder_path, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)

    return texts, labels


# ------------------------
# Preprocessing Function
# ------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


# ------------------------
# Main Pipeline
# ------------------------
def main():

    print("Loading data...")
    train_texts, train_labels = load_data('/Users/frangkysupit/Documents/aclImdb/train')
    test_texts, test_labels = load_data('/Users/frangkysupit/Documents/aclImdb/test')

    print("Preprocessing...")
    train_texts = [preprocess_text(t) for t in train_texts]
    test_texts = [preprocess_text(t) for t in test_texts]

    print("TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("Training model...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, train_labels)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(test_labels, y_pred))
    print("Precision:", precision_score(test_labels, y_pred))
    print("Recall:", recall_score(test_labels, y_pred))
    print("F1-score:", f1_score(test_labels, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))


if __name__ == "__main__":
    main()