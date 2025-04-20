# 📧 Spam-Ham Classification Using Machine Learning

This project implements a machine learning pipeline to classify SMS messages as **Spam** or **Ham** (Not Spam) using different classifiers and performance evaluation techniques.

## 🚀 Overview

The notebook loads a dataset of SMS messages labeled as spam or ham, cleans the data, and applies various machine learning models to classify the messages. The models' performance is then visualized using confusion matrices and evaluated with metrics like accuracy, precision, recall, and F1-score.

## 📊 Models Used

- Multinomial Naive Bayes (`MultinomialNB`)
- Support Vector Classifier (`SVC`)
- Random Forest Classifier (`RandomForestClassifier`)

## 🔍 Evaluation Metrics

Each model is evaluated based on:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (visualized using `ConfusionMatrixDisplay` from `sklearn.metrics`)

## 🧪 Dataset

The dataset used appears to be a labeled collection of SMS messages. Each message is tagged either as:
- **Ham** – Not Spam
- **Spam** – Unwanted promotional or scam messages

## 📦 Libraries & Dependencies

- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn` (Scikit-learn)

## 📈 Visualization

The confusion matrices for the classifiers are plotted in a 2x2 grid layout using `matplotlib` to show how well each classifier performs on the test data.

## 📌 How to Run

1. Install the required libraries using `pip install -r requirements.txt` (if provided).
2. Open the notebook `SpamHamClassification.ipynb` in Jupyter or any notebook environment.
3. Run all cells sequentially to train and evaluate the models.

## 🧠 Future Improvements

- Include more advanced NLP techniques like TF-IDF or word embeddings.
- Add deep learning models like LSTM for sequential text processing.
- Add cross-validation and hyperparameter tuning (GridSearchCV).

## 👨‍💻 Author

Aryan Pophali  
Third-year Engineering Student  
International Institute of Information Technology  
Passionate about Machine Learning & AI 🚀

---

📌 *Note: This project is part of a learning journey into text classification and supervised machine learning.*
