# 🤖 Amazon Alexa Sentiment Intelligence Engine & ML App

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-ff6600?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/NLP-NLTK%20%7C%20CountVectorizer-green?style=for-the-badge" alt="NLP">
</p>

---

## 📌 Executive Summary

An end-to-end Natural Language Processing (NLP) and Machine Learning system that extracts customer sentiment from verified consumer reviews. Features an interactive **Streamlit web application** and pre-trained ensemble models delivering sub-second real-time sentiment scoring.

Originally developed as part of the **CodeAlpha Data Analytics & ML Internship Program**.

---

## 🏗️ NLP & Modeling Pipeline

`
[Customer Review Text]
        │
        ▼
[Text Cleaning & Normalization]
  • Lowercasing & Punctuation Stripping
  • Tokenization & Stopword Elimination (NLTK)
  • WordNet Lemmatization
        │
        ▼
[Feature Extraction]
  • N-Gram CountVectorizer (Unigram & Bigram)
  • TF-IDF Statistical Weighting
  • Feature Scaling
        │
        ▼
[Model Inference & Ensemble Scoring]
  • Random Forest Classifier (Optimized Trees)
  • XGBoost Extreme Gradient Boosting
        │
        ▼
[Streamlit Interactive Dashboard]
  • Instant sentiment classification (Positive / Negative)
  • Probability distribution telemetry
`

---

## 📊 Benchmark Model Performance

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 87.4% | 85.9% | 89.2% | 0.875 |
| **Random Forest** | 92.1% | 91.4% | 93.3% | 0.923 |
| **XGBoost (Optimized)** | **94.8%** | **94.2%** | **95.6%** | **0.949** |

---

## 🚀 Quickstart & Running the App

### 1. Clone & Set Up
`ash
git clone https://github.com/Ayan-Rajput-Analysis/CodeAlpha_-Sentiment_Analysis.git
cd CodeAlpha_-Sentiment_Analysis/alexa_sentiment
pip install streamlit scikit-learn xgboost pandas numpy nltk
`

### 2. Launch the Streamlit App
`ash
streamlit run app.py
`

### 3. Pre-Trained Weights Included
The repository includes pre-fitted artifacts ready for instant production deployment:
- lexa_sentiment/countVectorizer.pkl
- lexa_sentiment/scaler.pkl
- lexa_sentiment/model_rf.pkl
- lexa_sentiment/model_xgb.pkl

---

## 👨‍💻 Author
**Ayan Rajput**  
- **GitHub:** [@Ayan-Rajput-Analysis](https://github.com/Ayan-Rajput-Analysis)  
- **LinkedIn:** [Ayan Rajput](https://www.linkedin.com/in/ayan-rajput-8781a7397/)