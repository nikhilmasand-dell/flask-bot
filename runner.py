from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobServiceClient
import openai
import pandas as pd
import PyPDF2
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import mammoth

app = Flask(__name__)

# Set up Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING = "KEYS"
BLOB_CONTAINER_NAME = "nikhilcontainer"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Azure OpenAI setup
openai.api_type = "azure"
openai.api_base = "KEYS"
openai.api_version = "2023-03-15-preview"
openai.api_key = "KEYS"
MODEL_DEPLOYMENT_NAME = "gpt-4"
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"

file_embeddings = []
uploaded_files = []

def clean_text(text):
    """Clean and normalize text."""
    return re.sub(r"[\r\n\t\f]+", " ", text).strip()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF files."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return clean_text(text)

def extract_text_from_docx(docx_file):
    """Extract text from DOCX files."""
    doc = Document(docx_file)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return clean_text(text)

def extract_text_from_doc(doc_file):
    """Extract text from DOC files using Mammoth."""
    try:
        with doc_file as file:
            result = mammoth.extract_raw_text(file)
        return clean_text(result.value)
    except Exception as e:
        return f"Error extracting text from .doc file: {e}"

def embed_text(text):
    """Embed text using Azure OpenAI's text-embedding-ada-002."""
    try:
        response = openai.Embedding.create(
            input=text,
            engine=EMBEDDING_DEPLOYMENT_NAME
        )
        return response['data'][0]['embedding']
    except Exception as e:
        return None

def split_into_chunks(text, chunk_size=500):
    """Split text into manageable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def search_embedded_content(query):
    """Search for the most relevant embedded content."""
    if not file_embeddings:
        return None, 0

    query_embedding = embed_text(query)
    if query_embedding is None:
        return None, 0

    similarities = [cosine_similarity([query_embedding], [item["embedding"]])[0][0]
                    for item in file_embeddings]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score > 0.75:  # Threshold for relevance
        return file_embeddings[best_match_idx]["content"], best_score
    return None, best_score

def enhance_answer_with_gpt(query, context):
    """Enhance the response using GPT-4 by combining query and relevant context."""
    try:
        response = openai.ChatCompletion.create(
            engine=MODEL_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "An error occurred while generating the response."

@app.route("/", methods=["GET", "POST"])
def index():
    global file_embeddings
    messages = [{"role": "system", "content": "You can ask questions or upload files for analysis."}]

    if request.method == "POST":
        user_message = request.form.get("message")
        if user_message:
            messages.append({"role": "user", "content": user_message})

            # Search for relevant content
            result, score = search_embedded_content(user_message)
            if result:
                enhanced_response = enhance_answer_with_gpt(user_message, result)
                messages.append({"role": "assistant", "content": enhanced_response})
            else:
                # Fallback to GPT-4 for general responses
                general_response = openai.ChatCompletion.create(
                    engine=MODEL_DEPLOYMENT_NAME,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )['choices'][0]['message']['content']
                messages.append({"role": "assistant", "content": general_response})

    return render_template("index.html", messages=messages)

@app.route("/upload", methods=["POST"])
def upload_file():
    global file_embeddings, uploaded_files

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    filename = secure_filename(file.filename)
    uploaded_files.append(filename)

    try:
        if file.content_type == "text/plain":
            file_content = file.read().decode("utf-8")
        elif file.content_type == "text/csv":
            df = pd.read_csv(file)
            file_content = df.to_csv(index=False)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file)
            file_content = df.to_csv(index=False)
        elif file.content_type == "application/pdf":
            file_content = extract_text_from_pdf(file)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_content = extract_text_from_docx(file)
        elif file.content_type == "application/msword":
            file_content = extract_text_from_doc(file)

        # Split and embed file content
        chunks = split_into_chunks(file_content)
        file_embeddings = [{"content": chunk, "embedding": embed_text(chunk)} for chunk in chunks if chunk.strip()]

        # Save file to Azure Blob Storage
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(file, overwrite=True)
        return jsonify({"message": f"File '{filename}' uploaded and embedded successfully."})

    except Exception as e:
        return jsonify({"error": f"Error processing file: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
