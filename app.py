"""
Multi-PDF RAG Chatbot with Persistent Memory
=============================================
All 8 problems fixed:
  1. create_retrieval_chain        → langchain_classic.chains (v1.x correct import)
  2. create_stuff_documents_chain  → langchain_classic.chains.combine_documents
  3. OllamaEmbeddings              → langchain_ollama (canonical, not deprecated)
  4. Ollama LLM                    → langchain_ollama.OllamaLLM (canonical)
  5. Chroma vectorstore            → langchain_chroma (canonical, not deprecated)
  6. Multi-PDF support             → loads ALL .pdf files in the current directory
  7. Conversation memory           → full chat_history passed on every turn
  8. Persistent vectorstore        → embeddings saved to disk; rebuilt only when
                                     PDFs change (fingerprinted by name + size)
"""

import os
import glob
import hashlib
import sys
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # runs in Python, no Ollama needed
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ──────────────────────────────────────────────
# CONFIG  (16 GB RAM optimised)
# ──────────────────────────────────────────────
CHROMA_DIR    = "./chroma_db"
HASH_FILE     = "./chroma_db/.pdf_hash"

# all-MiniLM-L6-v2: only ~90 MB RAM, runs in Python (no Ollama),
# so your LLM gets the full RAM budget.
EMBED_MODEL   = "all-MiniLM-L6-v2"

LLM_MODEL     = "ALIENTELLIGENCE/psychologistv2"
CHUNK_SIZE    = 800   # smaller chunks = less RAM per batch
CHUNK_OVERLAP = 100
TOP_K         = 3     # retrieve 3 chunks instead of 5 (saves context window RAM)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_pdf_paths() -> list[str]:
    """Return all PDF files in the current directory (sorted)."""
    pdfs = sorted(glob.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDF files found in the current directory.")
    return pdfs


def compute_pdf_hash(pdf_paths: list[str]) -> str:
    """
    Fingerprint = filename + file-size for every PDF.
    Any addition / removal / change triggers a vectorstore rebuild.
    """
    fingerprint = "|".join(f"{p}:{os.path.getsize(p)}" for p in pdf_paths)
    return hashlib.md5(fingerprint.encode()).hexdigest()


def load_and_split(pdf_paths: list[str]):
    """Load every PDF and split into overlapping chunks."""
    all_docs = []
    for path in pdf_paths:
        print(f"  • Loading: {os.path.basename(path)}")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  >> {len(chunks)} chunks created from {len(pdf_paths)} PDF(s)")
    return chunks


def needs_rebuild(pdf_paths: list[str]) -> bool:
    """Return True if vectorstore is missing or PDFs have changed."""
    if not os.path.isdir(CHROMA_DIR) or not os.path.isfile(HASH_FILE):
        return True
    with open(HASH_FILE) as f:
        saved = f.read().strip()
    return saved != compute_pdf_hash(pdf_paths)


def build_vectorstore(pdf_paths: list[str], embeddings) -> Chroma:
    """Create Chroma vectorstore from scratch and persist to disk."""
    print("\nBuilding vector database (first run takes a moment)...")
    chunks = load_and_split(pdf_paths)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(compute_pdf_hash(pdf_paths))

    print("  >> Vector database saved to disk.\n")
    return vectorstore


def load_vectorstore(embeddings) -> Chroma:
    """Load an existing persisted Chroma vectorstore."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# ──────────────────────────────────────────────
# PROMPT  (supports full chat history)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with deep expertise in the books provided. "
    "Use ONLY the retrieved context below to answer the user's question. "
    "If the answer is not found in the context, honestly say you don't know. "
    "Keep your answers clear, structured, and accurate. "
    "Use bullet points when listing items.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),   # ← full conversation history injected here
    ("human", "{input}"),
])


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Multi-PDF RAG Chatbot  (type 'exit' to quit)")
    print("=" * 55)

    # Discover all PDFs in the project directory
    pdf_paths = get_pdf_paths()
    print(f"\nFound {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  • {os.path.basename(p)}")

    # HuggingFaceEmbeddings runs purely in Python (no Ollama process).
    # all-MiniLM-L6-v2 uses only ~90 MB RAM — leaving RAM free for the LLM.
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},   # force CPU — safe on any machine
        encode_kwargs={"batch_size": 32}, # process 32 chunks at a time
    )
    print("  >> Embedding model ready.")

    # Vectorstore — reuse cached version if PDFs haven't changed
    if needs_rebuild(pdf_paths):
        vectorstore = build_vectorstore(pdf_paths, embeddings)
    else:
        print("\nLoading existing vector database from disk...")
        vectorstore = load_vectorstore(embeddings)
        print("  >> Loaded.\n")

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # LLM (runs locally via Ollama)
    # num_ctx: context window size — keep low to save RAM on 16 GB machines
    llm = OllamaLLM(model=LLM_MODEL, num_ctx=2048)

    # RAG chain
    qa_chain  = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # ── Chat loop with persistent in-session memory ──
    chat_history: list = []   # HumanMessage / AIMessage objects

    print("Ready! Ask anything about your PDFs.\n")
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        # --- Step 1: retrieve relevant chunks ---
        print("\n[Searching your PDFs...]")
        docs = retriever.invoke(user_query)

        # --- Step 2: stream the LLM response token by token ---
        # Build the prompt manually so we can stream
        formatted = prompt.format_messages(
            context="\n\n".join(d.page_content for d in docs),
            chat_history=chat_history,
            input=user_query,
        )

        print("Assistant: ", end="", flush=True)
        t_start = time.time()
        answer_tokens = []

        for chunk in llm.stream(formatted):
            print(chunk, end="", flush=True)   # print each token as it arrives
            answer_tokens.append(chunk)

        answer = "".join(answer_tokens)
        elapsed = time.time() - t_start
        print(f"\n\n[Done in {elapsed:.1f}s]")
        print("-" * 55)

        # Save this exchange to history
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()