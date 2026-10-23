# Multi-Agent Code Helper
An AI assistant that helps developers understand an unfamiliar Python codebase. Ask where something is implemented, what a file does, how modules connect, or how to get started—and receive answers grounded in the target repository.
Built with **Google ADK**, one coordinator, four specialist workers, a custom RAG pipeline, **NVIDIA NIM** embeddings, and **Pinecone** serverless storage. It supports ADK terminal and web interfaces; it does not include a custom website.
## 1. Architecture
```text
User
 |
Coordinator (routes requests; does not answer directly)
 |-- code_searcher       -> rag_search -> RAG / Pinecone
 |-- code_explainer      -> Filesystem MCP + read_project_file
 |-- architecture_mapper -> list_project_files + inspect_imports (Python AST)
 `-- onboarding_guide    -> get_project_overview (files + README)
```
Editable diagram: [`docs/architecture.drawio`](docs/architecture.drawio), open with [diagrams.net](https://app.diagrams.net).
The coordinator has no tools of its own and delegates to one worker. Workers cannot transfer tasks to peers or back to the coordinator, keeping routing to one coordinator-to-worker hop.
## 2. Technology Stack
| Component | Technology / configuration |
|---|---|
| Language | Python 3.13.7 (Python 3.11+ required) |
| Agent framework | Google ADK 2.12.0 |
| LLM gateway | LiteLLM 1.105.1 |
| Chat model | `meta/llama-3.2-11b-vision-instruct` via NVIDIA NIM |
| Embedding model | `nvidia/nemotron-3-embed-1b` via NVIDIA NIM |
| Vector database | Pinecone serverless, 2048 dimensions, cosine similarity |
| Filesystem tools | Node.js / npx + `@modelcontextprotocol/server-filesystem` |
| Environment | `python-dotenv` |
| Tests | `pytest` 9.1.1 |
The chat model is created once in `code_helper/config.py` and shared by all five agents. Embedding configuration is separate from chat configuration.
## 3. Project Layout
```text
demo/
├── code_helper/
│   ├── __init__.py             # ADK root_agent discovery
│   ├── agent.py                # Coordinator
│   ├── agents.py               # Four worker agents
│   ├── config.py               # Environment and model configuration
│   ├── ingestion.py            # RAG: safe file discovery and reading
│   ├── chunking.py             # Line-based chunking
│   ├── embeddings.py            # NVIDIA NIM embeddings
│   ├── vector_store.py          # Pinecone storage
│   ├── retrieval.py             # Similarity retrieval and formatting
│   ├── rag_pipeline.py          # RAG orchestration
│   ├── index_codebase.py        # Indexing CLI
│   ├── mcp_filesystem.py        # Restricted, read-only Filesystem MCP
│   ├── paths.py                 # Path-traversal protection
│   └── tools/                   # Agent tool implementations
├── target_project/              # Repository being analysed
├── tests/                       # Offline pytest suite
├── scripts/                     # Live integration checks
├── docs/                        # Architecture diagram and planning
├── requirements.txt
├── pytest.ini
├── ruff.toml
├── .env.example
└── README.md
```
The helper indexes `target_project/`, not its own source files.
## 4. RAG Pipeline
The custom RAG implementation does not use LangChain or LlamaIndex.
| Stage | Module | Purpose |
|---|---|---|
| 1. Ingestion | `ingestion.py` | Finds files; skips Git, virtual environments, caches, secrets, binaries, and oversized files |
| 2. Chunking | `chunking.py` | Splits code into overlapping line-based chunks |
| 3. Embeddings | `embeddings.py` | Creates NVIDIA NIM embeddings with batching, retries, and dimension checks |
| 4. Storage | `vector_store.py` | Stores vectors and file/line/text metadata in Pinecone |
| 5. Retrieval | `retrieval.py` | Embeds the query with the same model and returns top-k results with source details |
A literally named file (for example, "where is `main.py`?") is read directly rather than searched semantically, so it is always returned when it exists; a missing file is reported explicitly.
Default chunk size is **100 lines**, with **20 lines of overlap**. Configure these using `CHUNK_SIZE` and `CHUNK_OVERLAP`. Each chunk records its file, line range, stable ID, and original text. Each chunk is embedded with its relative path prepended (`# file: <path>`) so filename and module queries match the right file; re-run the indexer after changing this contract.
**Chunking limitations:** chunks may begin mid-function or split long functions; overlap only partly mitigates this. Whitespace-heavy or repetitive files may produce low-information or similar chunks. Overlap also increases embedding cost and index size.
## 5. Agents and Tools
| Agent | Responsibility | Main tool(s) |
|---|---|---|
| `coordinator` | Routes a question to one worker | Delegation only |
| `code_searcher` | Finds where code is implemented | `rag_search` |
| `code_explainer` | Explains a file's purpose and contents | Filesystem MCP, `read_project_file` |
| `architecture_mapper` | Lists files and analyses imports | `list_project_files`, `inspect_imports` |
| `onboarding_guide` | Explains purpose, setup, and reading order | `get_project_overview` |
The five custom Python tools are `rag_search`, `read_project_file`, `list_project_files`, `inspect_imports`, and `get_project_overview`. Four are deterministic and do not use AI — for example, `inspect_imports` parses code with Python's `ast` module. Internal settings are loaded by the tools rather than exposed as model-supplied arguments. Workers also enforce per-agent tool and model-call budgets (`MAX_TOOL_CALLS`, `MAX_LLM_CALLS`) so a run cannot loop indefinitely.
## 6. Filesystem MCP and Security
The `code_explainer` uses ADK's `McpToolset` and `StdioConnectionParams` to connect to the official Filesystem MCP server.
- MCP is restricted to the `target_project/` directory.
- Only allow-listed read tools are exposed; write, edit, move, and create operations are not exposed.
- The subprocess receives a sanitised environment that removes variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL`.
- If MCP is unavailable, `read_project_file` remains as a fallback. Set `DISABLE_MCP_FILESYSTEM=1` to disable MCP deliberately.
- Secrets belong in `.env`, which should never be committed. `.env.example` contains placeholders.
- Diagnostic configuration output is redacted. Missing keys produce a configuration error before network calls.
- Path traversal is guarded, and the vector store does not delete the index or vectors automatically.
## 7. Setup and Run
### Prerequisites
- Python 3.11+
- Node.js and `npx`
- NVIDIA NIM API key and Pinecone API key
### Install dependencies
```bash
pip install -r requirements.txt
```
### Configure environment
Run from `demo/`:
```powershell
Copy-Item .env.example .env
```
Add the required values to `.env`: `NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`), `NVIDIA_BASE_URL`, `NVIDIA_CHAT_MODEL`, `NVIDIA_EMBED_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and `PINECONE_REGION`.
Optional settings include `TARGET_REPO_PATH`, `EMBEDDING_DIMENSION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `PINECONE_NAMESPACE`, `MAX_LLM_CALLS`, `MAX_TOOL_CALLS`, `NVIDIA_MAX_TOKENS`, and `NVIDIA_TIMEOUT_SECONDS`.
### Index and verify the target project
```bash
python -m code_helper.index_codebase --progress
python -m code_helper.index_codebase --stats
```
### Run the agents
Run these commands from `demo/`, the directory containing `code_helper/`:
```bash
adk run code_helper
adk web
```
`adk run` provides a terminal interface; `adk web` starts the local web interface.
### Optional live integration checks
These require valid keys and network access:
```bash
python scripts/check_embeddings.py
python scripts/index_and_search.py
python scripts/mcp_list_tools.py
python scripts/run_agent_demo.py
```
## 8. Verification Results
The README records these results from the project environment:
- **Offline tests:** `python -m pytest` - **146 passed**. Tests are mocked and do not require keys or network access.
- **NVIDIA NIM embeddings:** embedding model verified; vector dimension **2048**.
- **Pinecone:** 6 chunks indexed; index dimension 2048 with cosine similarity. A test query about vector upsert ranked `vector_store.py` first.
- **Filesystem MCP:** server exposed 14 tools; all 10 allow-listed read-only tools were present, and a project file was read through MCP.
- **Agent routing:** the coordinator routed test questions to `code_searcher`, `code_explainer`, and `architecture_mapper`.
- **ADK terminal:** a query about the upsert implementation returned `vector_store.py`, lines 81–171.
- **ADK web:** server started locally and `/list-apps` returned HTTP 200 for `code_helper`.

### Measured latency
- **Retrieval (warm):** ~0.5–0.9 s per question (query embedding + Pinecone query).
- **First question per process:** ~4–5 s once, while the Pinecone client resolves the index host; the handle is reused afterwards.
- **Model calls dominate:** routing plus a tool-using worker means 2–3 sequential LLM calls. `NVIDIA_MAX_TOKENS` (default 800) bounds generation length; lower it for faster, shorter answers. `NVIDIA_TIMEOUT_SECONDS` (default 60) stops a stalled request from blocking the session indefinitely, enforced at the ADK layer so provider adapters cannot override it.
- **MCP cold start:** the first explainer call starts the `npx` filesystem server (a few seconds); later calls reuse it.

The four scripts above are live integration checks; the offline unit tests are mocked.
## 9. Issues Fixed
1. **String `top_k`:** the model sometimes supplied `"5"` as a string. The tool now converts it to an integer and returns a friendly error for invalid input; a regression test covers this.
2. **Leaked internal settings argument:** ADK exposed an internal `settings` parameter to the model. Public tools now accept only user-facing arguments and obtain configuration through `get_settings`.
3. **Worker-to-worker delegation:** workers could previously delegate to each other and produce confused answers. Transfer restrictions now enforce one-hop coordinator-to-worker routing.
4. **Filename queries could miss the file:** "where is `main.py`?" searched semantically, and the top-5 chunks could omit `main.py` entirely, leaving the model to speculate from imports. `rag_search` now resolves a literally named file directly and reports a missing file explicitly. Chunks are also embedded with their relative path (`# file: <path>`), so module-name queries match the right file.
5. **Absent topics returned confident chunks:** "Where is the Kubernetes deployment configured?" returned unrelated `main.py`/`config.py` chunks and let the model guess. `rag_search` now requires at least one retrieved chunk to mention a meaningful query term; otherwise it reports "No relevant evidence" and the searcher must answer "Not found in the indexed code." Routing also sends every "where is ...?" question to `code_searcher`, and the architecture mapper no longer inspects arbitrary files.
6. **Slow repeated retrieval:** every question built a fresh Pinecone client, which resolved the index host over the control plane (measured 8 s, sometimes 27 s with retries). Retrieval now reuses one Pinecone handle and one embedding client, memoises query embeddings, and caps chat output with `NVIDIA_MAX_TOKENS` (default 800). Warm retrieval is ~0.5–0.9 s.
7. **Unbounded chat requests:** a stalled NVIDIA NIM completion could block a session indefinitely (observed during latency testing). The shared model now enforces `NVIDIA_TIMEOUT_SECONDS` (default 60) with `asyncio.wait_for`, so the bound holds even when LiteLLM ignores its own `timeout`, and disables LiteLLM's hidden retries.
## 10. Example Questions
- **“Where is the upsert to Pinecone implemented?”** `code_searcher` uses RAG to locate the relevant function and return file/line references.
- **“What does `query_pipeline.py` do?”** `code_explainer` reads the file and explains its query, retrieval, context-building, and answer-generation flow.
- **“How do modules import and depend on one another?”** `architecture_mapper` uses deterministic Python AST analysis to inspect imports and entry points.
## 11. Known Limitations
- Line-based chunks may not preserve complete function boundaries.
- The language model may occasionally add a claim not supported by tool output; treat retrieved code and deterministic tool results as the ground truth.
- `adk web` requires a browser and uses the same agent package as `adk run`.
