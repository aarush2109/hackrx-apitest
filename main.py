import os
import fitz
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import google.generativeai as genai

nltk.download("punkt")
nltk.download("punkt_tab")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")

genai.configure(api_key=GEMINI_API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
index = None
sentences = None

app = FastAPI()

def extract_text_from_pdf(file_path):
    try:
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return " ".join(text_parts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")

def create_faiss_index(text):
    global index, sentences
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

def search(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
    return [sentences[i] for i in indices[0]]

def generate_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return (response.text or "").strip()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = extract_text_from_pdf(file_path)
        create_faiss_index(text)
        return {"message": "File processed successfully"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/query/")
async def query_file(query: str = Form(...)):
    if index is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet")
    context = " ".join(search(query))
    answer = generate_answer(context, query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
