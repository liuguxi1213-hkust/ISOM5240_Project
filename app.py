import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Customer Review Analysis",
    layout="centered"
)

device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(
        "text-classification",
        model="Gliubf/distilbert-sst2-sentiment",
        device=device
    )

sentiment_pipeline = load_sentiment_pipeline()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive"
}

def simple_summary(text):
    sentences = text.replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip() != ""]
    if len(sentences) == 0:
        return "No summary available."
    if len(sentences) == 1:
        return sentences[0] + "."
    return sentences[0] + ". " + sentences[1] + "."

st.title("Customer Review Intelligent Analysis System")

review = st.text_area(
    "Enter Customer Review",
    height=200
)

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment_result = sentiment_pipeline(review)[0]

        sentiment = label_map.get(
            sentiment_result["label"],
            sentiment_result["label"]
        )

        confidence = sentiment_result["score"]

        summary = simple_summary(review)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Sentiment")
        st.write(sentiment)

        st.subheader("Confidence Score")
        st.progress(float(confidence))
        st.write(round(confidence, 4))
