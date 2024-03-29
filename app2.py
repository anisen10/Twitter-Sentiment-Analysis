import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

# Set page configuration
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
# Load trained logistic regression model
lr_model = joblib.load('lr_model.pkl')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess the training data
column_names = ['Target', 'ID', 'Date', 'Flag', 'User', 'Text']
total_rows = sum(1 for line in open('training.1600000.processed.noemoticon.csv', encoding='latin1'))
nrows = 50000
skiprows = sorted(random.sample(range(1, total_rows), total_rows - nrows))
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin1', names=column_names, skiprows=skiprows)
df['Text'] = df['Text'].apply(lambda x: x.lower())
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['Text'] = df['Text'].apply(lambda x: word_tokenize(x))
df['Text'] = df['Text'].apply(lambda x: [word for word in x if word not in stop_words])
df['Text'] = df['Text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['Text'] = df['Text'].apply(lambda x: ' '.join(x).lower())

# Initialize and fit the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Text'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
lr_model.fit(X_train, y_train)

# Set background image
st.markdown(
    """
    <style>
    body {
        background-image: url("Backgfround2.jpg");
        background-size: cover;
        font-family: Arial, sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8); /* Input box background color with transparency */
        color: black; /* Input text color */
        font-size: 18px; /* Input text font size */
        border-radius: 10px; /* Input box border radius */
        border: none; /* Remove input box border */
        padding: 10px; /* Add padding */
    }
    .stButton > button {
        background-color: #008CBA; /* Button background color */
        color: white; /* Button text color */
        font-size: 20px; /* Button text font size */
        border-radius: 10px; /* Button border radius */
        padding: 10px 20px; /* Add padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to preprocess a single sentence
def preprocess_sentence(sentence):
    # Lowercase the sentence
    sentence = sentence.lower()
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens into a single string
    processed_sentence = ' '.join(tokens)
    return processed_sentence

# Streamlit app
def main():
 
    # Main title
    st.title("Twitter Sentiment Analysis")

    # Input box for user to enter a sentence
    sentence = st.text_input("Enter a sentence:")

    # Button to classify the sentence
    if st.button("Classify"):
        # Preprocess the sentence
        processed_sentence = preprocess_sentence(sentence)

        # Vectorize the preprocessed sentence using the same TfidfVectorizer
        X_test = tfidf_vectorizer.transform([processed_sentence])

        # Predict sentiment using logistic regression model
        lr_prediction = lr_model.predict(X_test)[0]

        # Map prediction to sentiment label
        sentiment_label = {0: 'Negative', 4: 'Positive'}

        # Display classification result
        st.write(f"The sentiment of the sentence '{sentence}' is: {sentiment_label[lr_prediction]}")

if __name__ == "__main__":
    main()
