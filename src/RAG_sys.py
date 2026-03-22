import pickle

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

import RAG_setup

RAG_setup.setup_environment()

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embd = RAG_setup.get_embeddings()
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embd)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 7})

with open("./Rag/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("./Rag/text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

qa_chain = RAG_setup.setup_llm_chain()


def retrieve(query, top_k=5):
    vector_docs = retriever.invoke(query)
    vector_texts = [doc.page_content for doc in vector_docs]

    tokenized_query = query.split()
    bm25_texts = bm25.get_top_n(tokenized_query, text_chunks, n=top_k)

    all_texts = []
    seen = set()
    for text in bm25_texts + vector_texts:
        if text not in seen:
            seen.add(text)
            all_texts.append(text)

    return all_texts[: top_k * 2]


def rerank(query, texts):
    if "reranker" not in globals():
        return texts[:3]

    pairs = [[query, text] for text in texts]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, texts), reverse=True)
    return [text for score, text in ranked[:3]]


def ask(question):
    texts = retrieve(question)
    top_texts = rerank(question, texts)

    if "qa_chain" in globals():
        context = "\n\n".join(top_texts)
        response = qa_chain.invoke({"context": context, "question": question})
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response
    else:
        return f"Found {len(top_texts)} relevant passages."


query_text = (
    "Which university does Hunter Jacobson studies mentioned in the resume? "
    "He is a student of human resource"
)
# print(ask(query_text))
