from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
from typing import List

# Initialize FastAPI
app = FastAPI()

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index and chunks
index = None
chunks = []

# Define the request body for the question
class QuestionRequest(BaseModel):
    question: str
    k: int = 3  # Default to top 3 results

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as pdf_file:
        for page_num in range(pdf_file.page_count):
            page = pdf_file[page_num]
            text += page.get_text()
    return text

# Load and index the document
def load_document_and_index(file_path: str):
    global chunks, index
    document = extract_text_from_pdf(file_path)
    chunks = document.split(". ")  # Split by sentence
    embeddings = embedding_model.encode(chunks)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

# Load the document and index it at startup
load_document_and_index('x.pdf')  # Make sure this PDF file exists in your directory

# Function to retrieve relevant chunks based on the question
def retrieve_relevant_chunks(question: str, k: int):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k)
    return [chunks[idx] for idx in indices[0]]

# Define the API route for answering questions
@app.post("/answer/")
async def get_answer(request: QuestionRequest):
    if index is None or len(chunks) == 0:
        raise HTTPException(status_code=500, detail="Document not loaded or indexed.")
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(request.question, request.k)
    
    # Return the relevant chunks
    return {"question": request.question, "relevant_chunks": relevant_chunks}
