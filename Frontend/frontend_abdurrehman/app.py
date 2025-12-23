# app.py
import os
import io
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Twitter Sentiment (CNN–LSTM)", layout="wide")
st.title("Twitter Sentiment Analysis – CNN–LSTM")
st.caption("Load saved model and tokenizer, classify single tweets or Excel files at scale.")

# -----------------------------
# Paths / constants
# -----------------------------
MODEL_PATH = "cnn_lstm_sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# IMPORTANT: must match your training setup
MAX_LEN = 50          # sequence length used during training
CLASS_NAMES = ["Negative", "Neutral", "Positive"]  # order must match your model's softmax

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, tok_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer file not found at: {tok_path}")

    model = load_model(model_path)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def basic_clean(text: str) -> str:
    """
    Minimal cleaning aligned with training:
    - lowercase, remove URLs, @mentions, hashtags (keep the word), non-letters (keep spaces)
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)            # URLs
    text = re.sub(r"@\w+", " ", text)                        # @mentions
    text = re.sub(r"#", " ", text)                           # remove '#', keep word
    text = re.sub(r"[^a-z\s]", " ", text)                    # keep letters/spaces only
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_texts(texts, tokenizer, max_len=MAX_LEN):
    cleaned = [basic_clean(t if isinstance(t, str) else "") for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    x = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return cleaned, x

def softmax_to_df(probs: np.ndarray, class_names=CLASS_NAMES) -> pd.DataFrame:
    prob_df = pd.DataFrame(probs, columns=[f"p({c})" for c in class_names])
    pred_idx = probs.argmax(axis=1)
    pred_label = [class_names[i] for i in pred_idx]
    prob_df.insert(0, "Predicted", pred_label)
    prob_df.insert(1, "Confidence", probs.max(axis=1).round(4))
    return prob_df

def predict_texts(model, tokenizer, texts: list):
    cleaned, X = preprocess_texts(texts, tokenizer, MAX_LEN)
    probs = model.predict(X, verbose=0)
    result_df = pd.DataFrame({"Original": texts, "Cleaned": cleaned})
    prob_df = softmax_to_df(probs, CLASS_NAMES)
    return pd.concat([result_df, prob_df], axis=1)

def make_download(df: pd.DataFrame, filename: str = "sentiment_results.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
with st.spinner("Loading model and tokenizer..."):
    try:
        model, tokenizer = load_artifacts(MODEL_PATH, TOKENIZER_PATH)
        st.success("Artifacts loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

# -----------------------------
# Sidebar: Mode Selection
# -----------------------------
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio(
    "Choose input type",
    ["Single Tweet", "Batch (Excel)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Model file:** `{MODEL_PATH}`")
st.sidebar.write(f"**Tokenizer:** `{TOKENIZER_PATH}`")
st.sidebar.write(f"**Max length:** `{MAX_LEN}`")
st.sidebar.write(f"**Classes:** {', '.join(CLASS_NAMES)}")

# -----------------------------
# Single Tweet Mode
# -----------------------------
if mode == "Single Tweet":
    st.subheader("Single Tweet Prediction")
    txt = st.text_area("Enter a tweet (short text):", height=120, placeholder="Type or paste a tweet here...")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        trigger = st.button("Predict", use_container_width=True)

    if trigger:
        if not txt.strip():
            st.warning("Please enter a non-empty tweet.")
        else:
            with st.spinner("Scoring..."):
                out_df = predict_texts(model, tokenizer, [txt])
            st.success("Done.")
            st.write("### Result")
            st.dataframe(out_df, use_container_width=True)
            make_download(out_df, "single_tweet_result.csv")

# -----------------------------
# Batch Mode (Excel)
# -----------------------------
else:
    st.subheader("Batch Prediction from Excel")
    uploaded = st.file_uploader("Upload an Excel file (.xlsx or .xls)", type=["xlsx", "xls"])

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded, engine="openpyxl" if uploaded.name.endswith("xlsx") else None)
        except Exception:
            # Fallback without engine hint
            df = pd.read_excel(uploaded)

        st.write("Preview of uploaded file:")
        st.dataframe(df.head(10), use_container_width=True)

        # Let user choose which column contains the text
        candidate_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not candidate_cols:
            st.error("No text (object) columns detected. Please upload a sheet with at least one text column.")
            st.stop()

        text_col = st.selectbox("Select the column containing tweets/text:", candidate_cols)

        min_rows = 1
        st.caption(f"Selected column: **{text_col}**. Rows detected: **{len(df)}**.")

        run_batch = st.button("Predict for all rows", type="primary", use_container_width=True)
        if run_batch:
            texts = df[text_col].fillna("").astype(str).tolist()
            if len(texts) < min_rows:
                st.warning("The file is empty or the selected column has no text.")
            else:
                with st.spinner("Running batch inference..."):
                    out_df = predict_texts(model, tokenizer, texts)
                st.success(f"Completed predictions for {len(texts)} rows.")

                # Combine with original data for full context
                merged = pd.concat([df.reset_index(drop=True), out_df], axis=1)

                st.write("### Batch Results (first 20 rows)")
                st.dataframe(merged.head(20), use_container_width=True)

                # Summary counts
                st.write("### Class Distribution")
                counts = merged["Predicted"].value_counts().reindex(CLASS_NAMES, fill_value=0)
                st.bar_chart(counts)

                make_download(merged, "batch_sentiment_results.csv")

# -----------------------------
# Footer / Help
# -----------------------------
with st.expander("ℹ️ Notes & Troubleshooting"):
    st.markdown(
        """
- The tokenizer and model must match the training pipeline (same vocabulary and MAX_LEN).
- If predictions seem off, confirm **CLASS_NAMES** order matches your model's softmax order.
- Excel column must contain the tweet text. Choose it in the selector.
- Cleaning here mirrors training assumptions: lowercasing, URL/@ removal, non-letters stripped.
        """
    )
