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

# --- PDF Upload and Save to Disk ---
pdf_folder = "pdfs"
os.makedirs(pdf_folder, exist_ok=True)

uploaded_pdf = st.file_uploader("Upload a PDF to add to the database", type=["pdf"])
if uploaded_pdf is not None:
    pdf_path = os.path.join(pdf_folder, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"Saved {uploaded_pdf.name} to disk.")

# --- Process All PDFs in Folder ---
if 'pdf_chunks' not in st.session_state:
    st.session_state['pdf_chunks'] = {}

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        print(f"Extracted text from {filename}:\n{text[:500]}")  # Debug: show first 500 chars
        if text.strip():
            chunks = chunk_text(text)
            print(f"Chunk count for {filename}: {len(chunks)}")  # Debug: show chunk count
            st.session_state['pdf_chunks'][filename] = chunks
            # Upsert chunks to Pinecone
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_chunk_{i}"
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[chunk]
                )
                embedding = response.data[0].embedding
                index.upsert([(doc_id, embedding)])
                print(f"Upserted chunk {i} for {filename}")  # Debug: confirm upsert

# --- Query Section ---
query = st.text_input("Enter your question:")

if query:
    query_embed = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    ).data[0].embedding

    top_k = 3
    results = index.query(vector=query_embed, top_k=top_k, include_metadata=False)

    if results.matches:
        context_text = ""
        for match in results.matches:
            match_id = match.id
            parts = match_id.split("_chunk_")
            if len(parts) != 2:
                continue
            pdf_name = parts[0]
            try:
                chunk_num = int(parts[1])
            except ValueError:
                continue
            if pdf_name in st.session_state['pdf_chunks']:
                chunks = st.session_state['pdf_chunks'][pdf_name]
            else:
                continue
            if chunk_num < len(chunks):
                context_text += chunks[chunk_num] + "\n---\n"
        print(f"Context sent to GPT-4:\n{context_text[:500]}")  # Debug: show first 500 chars
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