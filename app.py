import requests
import streamlit as st
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "facebook/bart-large-cnn"

API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def scrape_article(url):
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.find("h1")
    title = title.get_text(strip=True) if title else "No title"

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)

    return title, text


def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks


def summarize_article(text):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        payload = {
            "inputs": chunk,
            "parameters": {
                "max_length": 130,
                "min_length": 60,
                "do_sample": False
            }
        }

        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(response.text)

        summaries.append(response.json()[0]["summary_text"])

    return " ".join(summaries)


st.set_page_config(
    page_title="AI Article Summarizer",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

.card {
    background: #ffffff;
    padding: 32px;
    border-radius: 18px;
    box-shadow: 0 20px 40px rgba(15,23,42,.08);
    max-width: 700px;
    margin: auto;
}

.title {
    font-size: 32px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 6px;
}

.subtitle {
    color: #64748b;
    margin-bottom: 26px;
    font-size: 15px;
}

.summary-box {
    background: #f1f5f9;
    border-left: 5px solid #6366f1;
    padding: 18px;
    border-radius: 12px;
    color: #0f172a;
    line-height: 1.65;
}

input {
    background: #ffffff !important;
    color: #0f172a !important;
    border-radius: 10px !important;
    border: 1px solid #cbd5f5 !important;
}

button[kind="primary"] {
    background: linear-gradient(90deg,#6366f1,#4f46e5) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

with st.container():

    st.markdown('<div class="title">📰 AI Article Summarizer</div>', unsafe_allow_html=True)

    url = st.text_input("🔗 Enter article URL (CNN news site)")

    if st.button("Summarize"):
        if not url:
            st.warning("Enter article URL")
        else:
            with st.spinner("🔍 Scraping & summarizing..."):
                try:
                    title, article = scrape_article(url)

                    if len(article) < 100:
                        st.error("Cannot summarize articles with less than 100 words.")
                    else:
                        summary = summarize_article(article)

                        st.subheader("Article Title")
                        st.write(title)

                        st.subheader("Summary")
                        st.markdown(
                            f'<div class="summary-box">{summary}</div>',
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Error: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)
