import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Customer Review Analysis",
    page_icon="📊",
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

@st.cache_resource
def load_summarization_pipeline():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device
    )

sentiment_pipeline = load_sentiment_pipeline()
summarization_pipeline = load_summarization_pipeline()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive"
}

st.title("📊 Customer Review Intelligent Analysis System")
st.markdown("Analyze customer reviews using **Sentiment Analysis** and **Automatic Summarization**.")
st.divider()

review = st.text_area(
    "Enter Customer Review",
    placeholder="e.g. The product quality is excellent and delivery was fast...",
    height=200
)

if st.button("Analyze", type="primary"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):

            # Pipeline 1: Summarization
            word_count = len(review.split())
            if word_count >= 30:
                summary_result = summarization_pipeline(
                    review,
                    max_length=60,
                    min_length=20,
                    do_sample=False
                )
                summary = summary_result[0]["summary_text"]
            else:
                summary = review

            # Pipeline 2: Sentiment Analysis
            sentiment_result = sentiment_pipeline(review[:512])[0]
            sentiment = label_map.get(sentiment_result["label"], sentiment_result["label"])
            confidence = sentiment_result["score"]

        st.subheader("📝 Summary")
        st.info(summary)

        st.subheader("🎭 Sentiment")
        if sentiment == "Positive":
            st.success(f"✅ {sentiment}")
        else:
            st.error(f"❌ {sentiment}")

        st.subheader("📈 Confidence Score")
        st.progress(float(confidence))
        st.caption(f"Confidence: {round(confidence * 100, 2)}%")
