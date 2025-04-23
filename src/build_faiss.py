from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np


class CustomMedEmbed:
    def __init__(self):
        self.model = SentenceTransformer("abhinand/MedEmbed-small-v0.1")

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True)


def load_passages():
    dataset = load_dataset("enelpol/rag-mini-bioasq", name="text-corpus", split="test")
    docs = []
    for item in dataset:
        passage_id = item["id"]
        passage_text = item["passage"]
        docs.append(Document(page_content=passage_text, metadata={"source_id": passage_id}))
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks[:5]):
        print(f"[DEBUG] Chunk {i} metadata:", chunk.metadata)

    return chunks


def create_faiss_index(split_docs, save_path="../faiss_index"):
    embeddings = CustomMedEmbed()
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(save_path)
    return db


if __name__ == "__main__":
    print("Loading passages...")
    docs = load_passages()

    print("Splitting into chunks...")
    chunks = split_docs(docs)

    print("Building FAISS index...")
    index = create_faiss_index(chunks)

    print("Saved FAISS index.")