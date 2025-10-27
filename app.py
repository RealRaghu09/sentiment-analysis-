import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set[str](stopwords.words('english'))


# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Define prediction function
def predict_sentiment(text):
    clean_text = preprocess_text(text)
    vectorized = vectorizer.transform([clean_text]).toarray()
    pred = model.predict(vectorized)[0]

    label_map = {
        0: "😠 Negative",
        1: "🙁 Somewhat Negative",
        2: "😐 Neutral",
        3: "🙂 Somewhat Positive",
        4: "😄 Positive"
    }

    return label_map.get(pred, f"Unknown ({pred})")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Sentiment Analysis App")
st.markdown("Enter a sentence or review below to see its predicted sentiment.")

# Input text box
user_input = st.text_area("✍️ Enter your text here:", "", height=150)

if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
