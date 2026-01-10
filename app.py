import requests
from bs4 import BeautifulSoup
import streamlit as st
from transformers import pipeline, AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer)

def scrape_cnn_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find("h1")
    title = title.get_text() if title else "Judul tidak ditemukan"

    paragraphs = soup.find_all('p')

    article_text = ''
    for p in paragraphs:
        article_text += p.get_text() + ' '

    return title, article_text.strip()

def chunk_text_by_tokens(text, max_tokens=900):
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
    return chunks

def summarize(text, max_chunk=800):
    chunks = chunk_text_by_tokens(text)

    summaries = []
    
    words = text.split()
    
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=120,
            min_length=60,
            do_sample=False
        )
        
        summaries.append(summary[0]['summary_text'])
        
    return ' '.join(summaries)

st.set_page_config(page_title="Article Summarizer", layout="centered")
st.title("Article Summarizer")

url = st.text_input("URL CNN Article", "")

if st.button("Summarize"):
    if url:
        with st.spinner("Scraping and summarizing the article..."):
            try:
                title, date, article_text = scrape_cnn_article(url)
                
                if not article_text:
                    st.error("Could not extract article text. Please check the URL.")
                else:
                    summary = summarize(article_text)
                    
                    st.subheader("Article Title")
                    st.write(title)

                    st.subheader("Summary")
                    st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid URL.")
