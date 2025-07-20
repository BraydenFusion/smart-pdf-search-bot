import os
import openai
import pdfplumber
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "my-search-index"
index = pc.Index(index_name)

# --- PDF Chunking and Embedding ---
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

st.title("Smart PDF Search Bot")

query = st.text_input("Enter your question:")

if query:
    query_embed = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    ).data[0].embedding

    top_k = 3  # Number of top matches to retrieve
    results = index.query(vector=query_embed, top_k=top_k, include_metadata=False)

    if results.matches:
        context_text = ""
        folder_path = "pdfs"
        for match in results.matches:
            match_id = match.id
            pdf_name = match_id.split("_chunk_")[0]
            chunk_num = int(match_id.split("_chunk_")[1])
            pdf_path = os.path.join(folder_path, pdf_name)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            chunks = chunk_text(text)
            if chunk_num < len(chunks):
                context_text += chunks[chunk_num] + "\n---\n"

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer the question based on the context."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
            ]
        )
        st.markdown("**GPT-4 Answer:**")
        st.write(completion.choices[0].message.content)
    else:
        st.write("No matching document found.")