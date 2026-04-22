#  Lazy Reader

A local RAG (Retrieval-Augmented Generation) chatbot that lets you **chat with multiple PDF books at once** — entirely offline, no API keys needed.

## Features

- **Multi-PDF support** — drop any number of PDFs in the folder and they're all indexed automatically
- **Persistent vector database** — embeddings saved to disk, rebuilt only when PDFs change (fast startup after first run)
- **Full conversation memory** — the bot remembers everything you said earlier in the session
- **Streaming responses** — words appear token-by-token like ChatGPT, no blank staring
- **RAM optimised for 16 GB machines** — lightweight `all-MiniLM-L6-v2` embeddings (~90 MB) leave full RAM for the LLM
- **100% local** — runs via [Ollama](https://ollama.com), no internet required after setup

## Tech Stack

| Component | Library |
|-----------|---------|
| PDF loading | `langchain-community` PyPDFLoader |
| Text splitting | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 |
| Vector store | `chromadb` (persisted to disk) |
| LLM | Ollama (`ALIENTELLIGENCE/psychologistv2`) |
| RAG chain | `langchain-classic` |
| Memory | `langchain-core` HumanMessage / AIMessage |

## Setup

### 1. Install Ollama
Download from [https://ollama.com](https://ollama.com) and pull the model:
```bash
ollama pull ALIENTELLIGENCE/psychologistv2
```

### 2. Clone & install dependencies
```bash
git clone https://github.com/AdityaDesai7/Lazy-Reader.git
cd Lazy-Reader
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Add your PDFs
Drop any `.pdf` files into the project folder.

### 4. Run
```bash
python app.py
```

On first run it builds the vector database (takes a few minutes depending on PDF size).  
**Every run after that starts in seconds.**

## Usage

```
=======================================================
  Multi-PDF RAG Chatbot  (type 'exit' to quit)
=======================================================

Found 2 PDF(s):
  • Thinking, Fast and Slow.pdf
  • The Laws of Human Nature.pdf

Loading existing vector database from disk...
  >> Loaded.

Ready! Ask anything about your PDFs.

You: What is System 1 and System 2 thinking?

[Searching your PDFs...]
Assistant: System 1 is fast, automatic, and largely unconscious...

[Done in 18.3s]
-------------------------------------------------------
```

## Configuration

Edit these constants at the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `ALIENTELLIGENCE/psychologistv2` | Ollama LLM model |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `3` | Number of chunks retrieved per query |

## License

it's mine bro
