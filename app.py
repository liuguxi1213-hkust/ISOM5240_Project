import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    def summarize(text, max_length=60, min_length=20):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summarize

sentiment_pipeline = load_sentiment_pipeline()
summarize = load_summarization_pipeline()

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

            # Pipeline 1: Summarization (facebook/bart-large-cnn)
            word_count = len(review.split())
            if word_count >= 30:
                summary = summarize(review)
            else:
                summary = review

            # Pipeline 2: Sentiment Analysis (fine-tuned DistilBERT)
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
