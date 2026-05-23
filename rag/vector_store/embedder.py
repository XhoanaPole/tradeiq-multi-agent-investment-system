import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./rag/vector_store/db")
collection = chroma_client.get_or_create_collection(name="financial_docs")

# ── EMBED & STORE ────────────────────────────────────────
def embed_and_store(texts: list[str], ids: list[str], metadatas: list[dict] = None):
    """Embed a list of texts and store them in ChromaDB."""
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas or [{} for _ in texts]
    )
    print(f" Stored {len(texts)} documents in vector store.")

# ── RETRIEVE ─────────────────────────────────────────────
def retrieve(query: str, n_results: int = 3) -> list[str]:
    """Retrieve the most relevant documents for a query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]
    print(f" Retrieved {len(docs)} relevant documents.")
    return docs

# ── STORE NEWS HEADLINES ──────────────────────────────────
def store_news(ticker: str, headlines: list[dict]):
    """Embed and store news headlines for a ticker."""
    texts = [
        f"{h.get('title', '')} - {h.get('description', '')}"
        for h in headlines if h.get("title")
    ]
    if not texts:
        print(f" No headlines to store for {ticker}, skipping RAG storage.")
        return
    ids = [f"{ticker}-news-{i}" for i in range(len(texts))]
    metadatas = [{"ticker": ticker, "source": h.get("source", "")} for h in headlines if h.get("title")]
    embed_and_store(texts, ids, metadatas)

if __name__ == "__main__":
    # Quick test
    store_news("AAPL", [
        {"title": "Apple hits record high", "description": "Apple stock surged today.", "source": "Reuters"},
        {"title": "Apple launches new iPhone", "description": "New model expected this fall.", "source": "Bloomberg"}
    ])
    results = retrieve("Apple stock performance")
    for r in results:
        print(r)