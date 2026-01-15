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


st.set_page_config(page_title="AI Article Summarizer", layout="centered")
st.title("📰 AI Article Summarizer")

url = st.text_input("Enter article URL (CNN / news site)")

if st.button("Summarize"):
    if not url:
        st.warning("Masukkan URL dulu")
    else:
        with st.spinner("Scraping & summarizing..."):
            try:
                title, article = scrape_article(url)

                if len(article) < 100:
                    st.error("Cannot summarize articles with less than 100 words.")
                else:
                    summary = summarize_article(article)

                    st.subheader("Article Title")
                    st.write(title)

                    st.subheader("Summary")
                    st.write(summary)

            except Exception as e:
                st.error(f"Error: {e}")
