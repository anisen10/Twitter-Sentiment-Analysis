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

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
lr_model = joblib.load('lr_model.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['Text'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
lr_model.fit(X_train, y_train)

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

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    tokens = word_tokenize(sentence)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_sentence = ' '.join(tokens)
    return processed_sentence

def main():
 
    st.title("Twitter Sentiment Analysis")
    sentence = st.text_input("Enter a sentence:")

    if st.button("Classify"):
  
        processed_sentence = preprocess_sentence(sentence)
        X_test = tfidf_vectorizer.transform([processed_sentence])
        lr_prediction = lr_model.predict(X_test)[0]
        sentiment_label = {0: 'Negative', 4: 'Positive'}
        st.write(f"The sentiment of the sentence '{sentence}' is: {sentiment_label[lr_prediction]}")

if __name__ == "__main__":
    main()
