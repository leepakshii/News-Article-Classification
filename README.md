# News-Article-Classification
Overview

This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify news articles into predefined categories.
The goal is to automatically categorize text data (like politics, sports, tech, entertainment, etc.), which can be useful for information retrieval, recommendation systems, and content filtering.

Tech Stack

Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK

Models Used: Logistic Regression, Naive Bayes, Support Vector Machine (SVM)

 Project Structure
News-Article-Classification/
│── data/                  # Dataset (or link in README)
│── notebooks/
│     └── NewsArticleClassification.ipynb
│── reports/               # (Optional: PDF report/slides)
│── requirements.txt
│── README.md

 Key Steps

Data Preprocessing

Tokenization

Stopwords removal

Lemmatization / Stemming

Feature Engineering

Text vectorization (TF-IDF)

Model Training

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (SVM)

Evaluation

Accuracy

Precision, Recall, F1-score

Confusion Matrix

 Results

Logistic Regression → ~92% accuracy

Naive Bayes → ~89% accuracy

SVM → ~94% accuracy (Best performer)

▶How to Run

Clone this repo

git clone https://github.com/<your-username>/News-Article-Classification.git
cd News-Article-Classification


Install dependencies

pip install -r requirements.txt


Run the Jupyter notebook

jupyter notebook notebooks/NewsArticleClassification.ipynb
