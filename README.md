# Multi-Agent Canadian HTS Classification System (updated)

This project predicts candidate Canadian tariff classifications from item descriptions using a LangGraph workflow, optional Gemini or local Ollama models, and retrieval over a local tariff PDF.

> **Important:** AI output is only a research aid, not a binding customs ruling. Verify all codes against the current official Canadian Customs Tariff and obtain professional advice where appropriate.

## What changed in this update

- Removed the `langchain-community` dependency and replaced `PyPDFLoader` with `pypdf` plus `langchain_core.documents.Document`.
- Added bounded dependency ranges for a modern LangChain/LangGraph stack.
- Wired the search node into the workflow (the original graph defined it but did not execute it).
- Made missing Chroma index / missing Gemini key fail with clearer messages instead of crashing during import.
- Added `.env.example`, health endpoint, project-relative paths, and setup instructions.
- Kept the existing UI files and main scripts.

## Requirements

- Python **3.11 or 3.12** (recommended; avoid pre-release Python versions for this stack).
- For Gemini: Google AI Studio API key.
- For Ollama: Ollama installed and running, with a model pulled.
- The official Canadian tariff PDF, saved as `tariff.pdf` in this folder, for local RAG.

## Windows setup (Visual Studio / Command Prompt)

Open Command Prompt in this project folder:

```bat
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python launcher does not have 3.12, install Python 3.12 first and reopen the terminal. Do not use Python 3.15/pre-release builds for this project.

## Environment configuration

Copy the example environment file:

```bat
copy .env.example .env
notepad .env
```

Set `GEMINI_API_KEY` to your own key. Keep `.env` private; do not commit or share it. Optional model settings:

```env
HTS_GEMINI_MODEL=gemini-2.5-flash
HTS_OLLAMA_MODEL=qwen3:4b
OLLAMA_HOST=http://localhost:11434
HTS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Build the local tariff index

The repository ZIP you supplied did not contain `tariff.pdf`. Download the current official tariff PDF from the Government of Canada and place it in the project root, named `tariff.pdf`.

Then run:

```bat
python data_ingestion.py
```

To rebuild an existing index:

```bat
python data_ingestion.py --reset
```

The first embedding run downloads the sentence-transformer model and may take some time. The generated `chroma_db/` folder is local data and is not included in this source ZIP.

## Run the web application

```bat
python app.py
```

Open `http://127.0.0.1:8001` in your browser. Health check: `http://127.0.0.1:8001/api/health`.

The web application uses Gemini. If no API key is configured, it returns a clear configuration error.

## Run the command-line clients

Gemini:

```bat
python run_gemini.py
```

Ollama (after installing Ollama and pulling the model):

```bat
ollama pull qwen3:4b
ollama serve
```

In another terminal:

```bat
python run_ollama.py
```

## Batch evaluation

Put a `data.csv` file in the project root with columns named `desc` and `label`, then run:

```bat
python evaluate.py --model gemini-2.5-flash
```

or:

```bat
python evaluate.py --model qwen3:4b
```

The evaluation script writes a model-named CSV results file.

## Files

- `app.py` — FastAPI API and web UI host
- `graph.py` — LangGraph classification and escalation workflow
- `data_ingestion.py` — PDF parsing, chunking, embeddings, and Chroma index creation
- `run_gemini.py` / `run_ollama.py` — command-line entry points
- `evaluate.py` — dataset evaluation
- `static/` — frontend assets
- `.env.example` — environment variable template

## Troubleshooting

- **Terminal fails to start in Visual Studio:** use a normal Windows Command Prompt, then navigate to the project folder.
- **`tariff.pdf` missing:** add the official PDF before ingestion.
- **No RAG index:** run `python data_ingestion.py` after adding the PDF.
- **Gemini API key missing:** copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
- **Package resolver conflict:** use a fresh Python 3.11/3.12 virtual environment and install from `requirements.txt`.
- **Search is unavailable:** DDG results can be rate-limited or blocked. The workflow continues with the available context; verify tariff details against official sources.
