import re
import pickle
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@st.cache_resource
def download_nltk():
    for r in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(r, quiet=True)

download_nltk()

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

st.title("🔍 Fake News Detector")

news_input = st.text_area("Enter News Text:", height=200)

if st.button("Analyze"):

    if not news_input.strip():
        st.warning("⚠️ Enter text")
    else:
        cleaned = clean_text(news_input)
        features = vectorizer.transform([cleaned])

        probs = model.predict_proba(features)[0]

        
        threshold = 0.6

        if probs[1] > threshold:
            prediction = 1
        else:
            prediction = 0

        confidence = probs[prediction] * 100

        # Output
        if prediction == 0:
            st.success(f"✅ Real News ({confidence:.2f}%)")
        else:
            st.error(f"🚨 Fake News ({confidence:.2f}%)")

        # Debug info
        st.subheader("Confidence Breakdown")
        st.write(f"Real: {probs[0]*100:.2f}%")
        st.write(f"Fake: {probs[1]*100:.2f}%")