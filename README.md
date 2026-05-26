# Customer Review Intelligent Analysis System

## Project Overview

This project is an NLP-based web application developed using Streamlit and Hugging Face Transformers.

The system analyzes customer reviews and provides:
- Sentiment Classification
- Confidence Score
- Automatic Review Summary

The sentiment analysis model was fine-tuned using the SST2 dataset with DistilBERT.

---

## Features

### Sentiment Analysis
Predicts whether a customer review is:
- Positive
- Negative

### Confidence Score
Displays the prediction confidence level.

### Review Summarization
Generates a short summary of the customer review.

### Interactive Web Interface
Built using Streamlit for real-time interaction.

---

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- DistilBERT
- NLP
- Machine Learning

---

## Model Information

Base Model:
- distilbert-base-uncased

Dataset:
- Stanford Sentiment Treebank (SST2)

Fine-tuning Task:
- Binary Sentiment Classification

---

## Project Structure

```bash
ISOM5240_Project/
│
├── app.py
├── requirements.txt
├── README.md
└── distilbert-sst2-sentiment-final/
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Input

```text
The product quality is excellent and delivery was fast.
```

Example Output:
- Sentiment: Positive
- Confidence Score: 0.998

---

## Author

Guxi Liu

MSBA Student  
HKUST