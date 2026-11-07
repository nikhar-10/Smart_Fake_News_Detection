
# Smart Fake News Detection System

An AI and Machine Learning based web application that classifies news articles as FAKE or REAL using Natural Language Processing (NLP) and Machine Learning.

## 1. Project Overview

The Smart Fake News Detection System is designed to analyze news article text and predict whether the given article belongs to the FAKE or REAL category.

The system uses TF-IDF for text feature extraction and Logistic Regression for classification.

A Flask-based web interface allows users to submit news articles and view predictions.

## 2. Objectives

- Detect potentially fake news articles using Machine Learning.
- Apply Natural Language Processing techniques.
- Extract meaningful text features using TF-IDF.
- Train and evaluate a classification model.
- Provide a simple and interactive web interface.
- Store prediction history using SQLite.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| NLTK | NLP Library |
| TF-IDF | Text Feature Extraction |
| Logistic Regression | Classification |
| Flask | Web Framework |
| SQLite | Database |
| HTML & CSS | Frontend |
| Joblib | Model Serialization |

## 4. Dataset

Dataset: Fake and Real News Dataset

Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset files:
- Fake.csv
- True.csv

After preprocessing:
- Total Articles: 44,898
- Fake Articles: 23,481
- Real Articles: 21,417

Labels:
- 0 = FAKE
- 1 = REAL

## 5. Machine Learning Workflow

1. Dataset Collection
2. Data Preprocessing
3. Text Cleaning and Preparation
4. TF-IDF Feature Extraction
5. Train-Test Split
6. Logistic Regression Model Training
7. Model Evaluation
8. Model Saving
9. Flask Web Application Integration

## 6. Model Performance

The initial model training run achieved:

- Accuracy: Approximately 98.88%
- Precision: Approximately 99%
- Recall: Approximately 99%
- F1-Score: Approximately 99%

These results are based on the project's 80:20 train-test split.

## 7. Project Structure

Smart_Fake_News_Detection/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── news.csv
│
├── notebooks/
├── outputs/
│   ├── fake_news_model.pkl
│   └── news_history.db
│
├── screenshots/
│
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── database.py
│
├── templates/
│   ├── index.html
│   └── history.html
│
├── static/
├── venv/
├── app.py
├── requirements.txt
└── README.md

## 8. Installation and Execution

### Step 1: Clone the Repository

git clone YOUR_GITHUB_REPOSITORY_URL

### Step 2: Navigate to Project

cd Smart_Fake_News_Detection

### Step 3: Create Virtual Environment

python -m venv venv

### Step 4: Activate Environment

Windows:

venv\Scripts\activate

### Step 5: Install Dependencies

pip install -r requirements.txt

### Step 6: Prepare Dataset

python src/prepare_data.py

### Step 7: Train Model

python src/train_model.py

### Step 8: Run Application

python app.py

Open in browser:

http://127.0.0.1:5000

## 9. Main Features

- News Article Classification
- FAKE and REAL Prediction
- Model Confidence Score
- Input Validation
- Prediction History
- SQLite Database Integration
- Interactive Web Interface

## 10. Disclaimer

This project is an educational Machine Learning application.

Its predictions are based on patterns learned from the training dataset. A FAKE or REAL prediction does not establish the factual accuracy of an article.

The model should not be treated as a replacement for professional fact-checking or verification from reliable sources.

## 11. Future Enhancements

- Deep Learning-based classification
- Transformer models such as BERT
- News source credibility analysis
- Multilingual news detection
- Explainable AI integration
- Improved model generalization
- Deployment on cloud platforms

## 12. Project Category

Major Project | Artificial Intelligence & Machine Learning

**Project Name:** Smart Fake News Detection System
