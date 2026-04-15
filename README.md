# 🎬 Sentiment Analysis on Movie Reviews

This project implements a Natural Language Processing (NLP) pipeline to classify movie reviews as **positive** or **negative** using machine learning.

## 🚀 Features
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Model evaluation (Accuracy, Precision, Recall, F1-score)
- Confusion matrix visualization
- Dataset size vs accuracy analysis

## 🧠 Technologies Used
- Python
- NLTK
- Scikit-learn
- NumPy
- Matplotlib

## 📂 Project Structure
├── main.py # Main NLP pipeline
├── matrix.py # Confusion matrix visualization
├── acc_dsize.py # Dataset size vs accuracy plot
├── README.md
└── requirements.txt

## 📊 Dataset
- Stanford Large Movie Review Dataset (IMDb)
- 50,000 labeled reviews (balanced dataset)

## ⚙️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt