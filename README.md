# Spam-vs.-Ham-Classifier

Project Overview
This project focuses on building a text classification model to detect whether an incoming SMS or email message is spam or ham (not spam). It uses Natural Language Processing (NLP) for text preprocessing and a machine learning algorithm like Naive Bayes or Logistic Regression for classification.
Tools & Technologies Used
- Python
- NLTK (Natural Language Toolkit) for text preprocessing
- Scikit-learn for machine learning and model evaluation
- Dataset: SMS Spam Collection Dataset
Dataset Description
The SMS Spam Collection dataset contains a set of SMS messages tagged as either 'spam' or 'ham'. It is commonly used for text classification tasks and is publicly available from the UCI Machine Learning Repository.
Step-by-Step Implementation
1.	1. Load the Dataset
•	   - Read the dataset into a pandas DataFrame.
2.	2. Preprocess the Text
•	   - Convert to lowercase, remove punctuation, stopwords, and tokenize.
3.	3. Feature Extraction
•	   - Convert text to numerical features using TF-IDF or CountVectorizer.
4.	4. Train the Model
•	   - Use Naive Bayes or Logistic Regression for classification.
5.	5. Evaluate the Model
•	   - Measure accuracy, precision, recall, and F1 score.
6.	6. Predict New Messages
•	   - Use the trained model to classify new incoming messages.
Optional Enhancements
- Deploy the model as a REST API using Flask or FastAPI.
- Build a simple web UI for users to input text and get predictions.
- Use deep learning models like LSTM or BERT for improved accuracy.
Learning Outcomes
- Understanding of NLP preprocessing techniques.
- Application of text classification using machine learning.
- Evaluation and validation of ML models.

Distribution of Spam and Ham Messages:
<img width="549" height="393" alt="Distribution of Spam and Ham Messages" src="https://github.com/user-attachments/assets/30519284-95ca-4c6c-9f87-70f5eb57e126" />
