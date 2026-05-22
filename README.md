# Phishing Email Detector using Scikit-learn

## Task Description
Developed a machine learning-based Phishing Email Detector that classifies emails as phishing or safe using Scikit-learn and Natural Language Processing (NLP) techniques.

## Features
- Detects phishing and safe emails
- Uses Machine Learning classification
- Trains on a phishing email dataset
- Supports multi-line email input
- Displays prediction results in real-time
- Shows model accuracy
- Displays confusion matrix
- Beginner-friendly phishing detection system

## Technologies Used
- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Multinomial Naive Bayes

## Machine Learning Concepts Used
- Text Vectorization
- Feature Extraction
- Naive Bayes Classification
- Train-Test Split
- Accuracy Evaluation
- Confusion Matrix

## How It Works
1. A dataset containing phishing and safe emails is created.
2. Email text is converted into numerical features using CountVectorizer.
3. The dataset is split into training and testing data.
4. A Multinomial Naive Bayes model is trained using Scikit-learn.
5. Users can paste email content for phishing detection.
6. The model predicts whether the email is phishing or safe.

## Output
The application displays:
- Prediction result
- Model accuracy
- Confusion matrix
- Email classification

## Example Predictions

### Phishing Email
"Verify your bank account immediately"

Prediction:
phishing

### Safe Email
"Meeting scheduled for tomorrow"

Prediction:
safe

## Note
This project is created for educational and cybersecurity learning purposes only.
