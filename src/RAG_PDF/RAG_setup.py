import os
import pickle
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from transformers import pipeline


def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    load_dotenv(".env.RAG")
    hf_token = os.getenv("HF_TOKEN")
    return hf_token


def get_embeddings():
    embd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embd


def load_documents(data_dir="./Data"):
    loader = DirectoryLoader(data_dir, glob="**/*")
    docs = loader.load()
    return docs


def cleaner_func(docs):
    if isinstance(docs, list):
        cleared = []
        for doc in docs:
            if isinstance(doc, str):
                doc = re.sub(r"\n+", "\n", doc)
                doc = re.sub(r"\s+", " ", doc)
                cleared.append(doc.strip())
            elif hasattr(doc, "page_content"):
                content = doc.page_content
                content = re.sub(r"\n+", "\n", content)
                content = re.sub(r"\s+", " ", content)
                doc.page_content = content.strip()
                cleared.append(doc)
            else:
                cleared.append(doc)
        return cleared
    elif isinstance(docs, str):
        docs = re.sub(r"\n+", "\n", docs)
        docs = re.sub(r"\s+", " ", docs)
        return docs.strip()
    return docs


def setup_llm_chain():
    pipline = pipeline(
        "text-generation", model="microsoft/phi-2", max_new_tokens=512, temperature=0.3
    )
    llm = HuggingFacePipeline(pipeline=pipline)
    template = """Given the following context, answer the question accurately.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = prompt | llm
    return qa_chain


def split_documents(cleaned_docs):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    cleaned_chunk = text_split.split_documents(cleaned_docs)
    return cleaned_chunk


def create_vector_store(cleaned_chunk, embd, persist_dir="chroma_db"):
    vector_store = Chroma.from_documents(
        documents=cleaned_chunk, embedding=embd, persist_directory=persist_dir
    )
    return vector_store


def create_retriever(vector_store, k=7):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return retriever


def create_bm25_index(cleaned_chunk):
    text_chunks = [chunk.page_content for chunk in cleaned_chunk]
    tokenized_chunks = [chunk.split() for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, text_chunks


def save_bm25_index(bm25, text_chunks, cleaned_chunk, save_dir="./Rag"):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{save_dir}/text_chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)
    print(f"BM25 index saved to {save_dir}/bm25.pkl")

    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(cleaned_chunk, f)


def run_setup():
    setup_environment()
    embd = get_embeddings()

    docs = load_documents("./Data/Resume")
    cleaned_docs = cleaner_func(docs)
    cleaned_chunk = split_documents(cleaned_docs)

    create_vector_store(cleaned_chunk, embd)

    bm25, text_chunks = create_bm25_index(cleaned_chunk)
    save_bm25_index(bm25, text_chunks, cleaned_chunk)


if __name__ == "__main__":
    run_setup()
