# Multi-Agent Code Helper

An AI assistant that helps a developer understand an unfamiliar Python codebase. Ask it
*where* something is implemented, *what* a file does, *how* modules connect, or *how to
get started*, and it answers from the actual repository.

The system is built on **Google ADK** with **one coordinator and four specialist
workers**, backed by a **custom RAG pipeline** (no LangChain, no LlamaIndex) that uses
**NVIDIA NIM** embeddings and **Pinecone** serverless storage. It runs through the ADK
terminal (`adk run`) and web (`adk web`) interfaces — there is no custom website.

---

## 1. Architecture

```
                         ┌────────────────────────┐
            user ───────▶│      coordinator        │  (Boss)
                         │  routes, never answers  │
                         └───────────┬────────────┘
             ┌───────────────┬───────┴────────┬─────────────────┐
             ▼               ▼                ▼                 ▼
      ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐ ┌────────────────┐
      │code_searcher│ │code_explainer│ │architecture_mapper│ │onboarding_guide│
      │  rag_search │ │ MCP + read  │ │ list_project_files│ │ get_project_   │
      │             │ │ _project_file│ │ inspect_imports   │ │ overview       │
      └──────┬──────┘ └──────┬──────┘ └────────┬─────────┘ └───────┬────────┘
             │               │                 │                   │
        Pinecone        Filesystem MCP     ast (no AI)        files + README
```

The coordinator has **no tools of its own** and cannot answer technical questions
directly. It can only delegate, which keeps every answer traceable to a worker and its
tools. Workers are forbidden from handing off to each other
(`disallow_transfer_to_parent` / `disallow_transfer_to_peers`), so routing is always a
single `coordinator → worker` hop.

---

## 2. Technology stack (verified in this environment)

| Component | Required / used | Verified in this repo |
|---|---|---|
| Language | Python 3.13.7 | ✅ |
| Agent framework | `google-adk` | 2.12.0 |
| LLM gateway | LiteLLM | 1.105.1 |
| Chat model | `meta/llama-3.2-11b-vision-instruct` via `nvidia_nim/…` | used by all 5 agents |
| Embedding model | `nvidia/nemotron-3-embed-1b` via `nvidia_nim/…` | ✅ live dimension = **2048** |
| Vector DB | Pinecone serverless, 2048-dim, cosine, aws/us-east-1 | ✅ index `code-helper` |
| Filesystem | Node.js v24.21.0 + `npx` 11.19.0, `@modelcontextprotocol/server-filesystem` | ✅ |
| Env loading | `python-dotenv` | 1.2.4 |
| Tests | `pytest` | 9.1.1 |

The chat model is created **once** (`code_helper/config.py::get_chat_model`) and shared by
all five agents. Embedding configuration is kept **separate** and never reused for chat.

---

## 3. Project layout

```
demo/
├── code_helper/                 # the helper implementation (the deliverable)
│   ├── __init__.py              # exposes root_agent for ADK discovery
│   ├── agent.py                 # coordinator (root_agent)
│   ├── agents.py                # the four worker agents
│   ├── config.py                # env loading, validation, shared chat model
│   ├── ingestion.py             # RAG stage 1 – file discovery + safe reading
│   ├── chunking.py              # RAG stage 2 – chunk_code()
│   ├── embeddings.py            # RAG stage 3 – NVIDIA NIM embeddings
│   ├── vector_store.py          # RAG stage 4 – Pinecone
│   ├── retrieval.py             # RAG stage 5 – retrieve()/format_results()
│   ├── rag_pipeline.py          # orchestrates stages 1–4 (+ answer_question)
│   ├── index_codebase.py        # CLI: python -m code_helper.index_codebase
│   ├── mcp_filesystem.py        # Filesystem MCP wiring (restricted, read-only)
│   ├── paths.py                 # path-traversal guard
│   └── tools/
│       ├── search_tools.py      # rag_search
│       ├── explain_tools.py     # read_project_file
│       ├── architecture_tools.py# list_project_files, inspect_imports
│       └── onboarding_tools.py  # get_project_overview
├── target_project/              # the code under analysis (separate from the helper)
├── tests/                       # offline pytest suite
├── scripts/                     # live integration checks (need real keys)
├── requirements.txt
├── pytest.ini
├── ruff.toml                    # shared lint/format configuration
├── .gitignore
├── .env.example                 # placeholders only
└── README.md
```

`target_project/` holds the repository the agents analyse; the helper never mixes its own
source into the index.

---

## 4. The RAG pipeline (implemented by hand)

Each stage is its own module. The pipeline is called by the `code_searcher` tool and by
`python -m code_helper.index_codebase`.

| Stage | Module | What it does |
|---|---|---|
| 1. Ingestion | `ingestion.py` | Walks the target, skips `.git`, `.venv`, caches, secrets, binary and oversized files; preserves the original text and relative path. |
| 2. Chunking | `chunking.py` | `chunk_code()` — a custom line-based sliding window. |
| 3. Embeddings | `embeddings.py` | `nvidia/nemotron-3-embed-1b` via LiteLLM; batching, retries, dimension validation. |
| 4. Storage | `vector_store.py` | Pinecone serverless index, deterministic vector IDs, metadata (file + lines + text). |
| 5. Retrieval | `retrieval.py` | `retrieve()` — embeds the query with the **same** model and returns top-k with provenance. |

### Chunking parameters
- **chunk size = 100 lines**, **overlap = 20 lines** (starting values; configurable via
  `CHUNK_SIZE` / `CHUNK_OVERLAP`). Each chunk carries a stable id `file:start-end`, the
  relative filename, the start and end line, and the original text.

### Limitations of line-based chunking (honest notes)
- A chunk can begin mid-function, so a retrieved snippet is a *pointer* into the file, not
  a self-contained unit.
- A long function may be split across chunks; overlap reduces but does not remove this.
- Whitespace-heavy files yield low-information chunks; repetitive files yield
  near-duplicate chunks (de-duplicated within a file).
- Overlap increases both embedding cost and index size.
- Because the target files here are short, several whole files become a single chunk, so
  similarity scores between different files can be close. The `upsert` question still
  ranks `vector_store.py` first, but scores are not sharply separated.

---

## 5. Agents and their tools

| Agent | Type | Responsibility | Tools |
|---|---|---|---|
| `coordinator` | root (Boss) | Route a question to one worker and return its answer. | *(none — delegates only)* |
| `code_searcher` | worker | Where is X implemented? | `rag_search` |
| `code_explainer` | worker | What does a whole file do? | Filesystem **MCP** toolset + `read_project_file` |
| `architecture_mapper` | worker | Imports, dependencies, entry points. | `list_project_files`, `inspect_imports` |
| `onboarding_guide` | worker | Purpose, setup, reading order. | `get_project_overview` |

### The five custom tools (all plain Python, typed, documented, defensive)

| Tool | Signature | AI used? |
|---|---|---|
| `rag_search` | `rag_search(query: str, top_k: int = 5) -> str` | yes (embeddings + LLM reasoning) |
| `read_project_file` | `read_project_file(file_path: str) -> str` | **no** |
| `list_project_files` | `list_project_files() -> str` | **no** |
| `inspect_imports` | `inspect_imports(file_path: str) -> str` | **no** (`ast` only) |
| `get_project_overview` | `get_project_overview() -> str` | **no** |

The last three (plus `read_project_file`) are fully deterministic, satisfying the
"at least one tool must operate without AI" requirement — in fact four do. The MCP
server's built-in tools are **not** counted toward this minimum.

Tool functions expose only user-facing arguments. Internal configuration is resolved via a
module-level `get_settings` seam, because ADK turns every parameter into something the
model must fill in (this was a real bug — see §9).

---

## 6. Filesystem MCP

The `code_explainer` connects to the official Filesystem MCP server through ADK's
supported `McpToolset` (note: `MCPToolset` is deprecated in ADK 2.12) using
`StdioConnectionParams`:

```python
McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(target_root)],
            env=safe_subprocess_env(),
        ),
        timeout=30.0,
    ),
    tool_filter=list(READ_ONLY_MCP_TOOLS),
)
```

Restrictions:
- **Root**: only the target project directory is passed. Never a drive root, never the
  `.env` location.
- **Read-only**: a `tool_filter` exposes only read tools (`read_text_file`,
  `list_directory`, `search_files`, …). `write_file`, `edit_file`, `move_file`,
  `create_directory` are **not** surfaced.
- **Sanitised environment**: the `npx` child process receives only an allow-list of
  variables; anything whose name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or
  `CREDENTIAL` is removed, so the MCP server cannot read our API keys.
- If MCP is unavailable, the agent still works via `read_project_file`
  (set `DISABLE_MCP_FILESYSTEM=1` to skip MCP deliberately).

The tool names above were **verified against the live server**, not guessed (see §8).

---

## 7. Setup and running

### Prerequisites
- Python 3.11+ (verified on 3.13.7)
- Node.js + `npx` (verified on Node 24 / npx 11)
- An NVIDIA NIM API key and a Pinecone API key

### Install
```bash
pip install -r requirements.txt
```

### Configure
Copy the template and fill in your keys (never commit `.env`):
```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```
Required variables: `NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`), `NVIDIA_BASE_URL`,
`NVIDIA_CHAT_MODEL`, `NVIDIA_EMBED_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`PINECONE_CLOUD`, `PINECONE_REGION` (plus optional `TARGET_REPO_PATH`,
`EMBEDDING_DIMENSION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `PINECONE_NAMESPACE`).

### Index the target project (once)
```bash
python -m code_helper.index_codebase --progress
python -m code_helper.index_codebase --stats      # verify vector count
```

### Run the agents
```bash
adk run code_helper        # terminal, run from the directory containing code_helper/
adk web                    # web UI, run from the same directory
```
> Both commands must be run from `demo/` (the folder that contains `code_helper/`), because
> that is where ADK discovers `root_agent`.

### Live integration checks (optional, need real keys/network)
```bash
python scripts/check_embeddings.py     # verifies the embedding model + dimension
python scripts/index_and_search.py     # indexes + retrieves against real Pinecone
python scripts/mcp_list_tools.py       # lists the real MCP tools, reads a file via MCP
python scripts/run_agent_demo.py       # runs the coordinator and prints routing
```

---

## 8. Verified results (real runs, not claims)

**Offline unit tests** — deterministic, mocked, no keys/network:
```
$ python -m pytest
118 passed
```

**NVIDIA NIM embeddings** (`scripts/check_embeddings.py`):
```
Configured embedding model: nvidia_nim/nvidia/nemotron-3-embed-1b
Verified dimension        : 2048
First coordinates         : [0.5589, 0.0271, -0.1868, -0.0474]
EMBEDDING CHECK PASSED
```

**Pinecone indexing + retrieval** (`scripts/index_and_search.py`):
```
chunks_created: 6   chunks_indexed: 6   embedding_dimension: 2048
Index stats: { total_vector_count: 6, dimension: 2048, metric: "cosine",
               namespaces: { "code-helper": 6 } }

QUERY: Where is the upsert to the vector database implemented?
  [1] vector_store.py lines 81-171  score=0.7603
  [2] vector_store.py lines 1-100   score=0.7377
```

**Filesystem MCP** (`scripts/mcp_list_tools.py`):
```
Server exposes 14 tools; our 10 allow-listed read-only tools all exist.
Allowed directories: E:\assignment\demo\target_project
Read 2702 characters from target_project/config.py via MCP.
```

**Agent routing** (`scripts/run_agent_demo.py`) — `ROUTING: coordinator -> <worker>`
(re-verified after the worker-transfer fix):
```
coordinator -> code_searcher        (cited vector_store.py lines 81-171)
coordinator -> code_explainer       (explained query_pipeline.py end to end; used MCP)
coordinator -> architecture_mapper  (parsed real imports per module)
```

**ADK terminal interface** (`adk run code_helper`, run from `demo/`):
```
Running agent coordinator, type exit to exit.
[user]: Where is the upsert function implemented?
... LiteLLM completion() model= meta/llama-3.2-11b-vision-instruct; provider = nvidia_nim
[code_searcher]: The upsert function is implemented in the vector_store.py file,
                 lines 81-171, with a similarity score of 0.7294.
```
(Piping a single line makes ADK abort the session at EOF, which is expected.)

**ADK web interface** (`adk web`, run from `demo/`):
```
+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
| For local testing, access at http://127.0.0.1:8765.                         |
+-----------------------------------------------------------------------------+
GET /list-apps -> HTTP 200: ["code_helper"]
```

The offline tests are clearly **mocked** (`tests/`); the four scripts above are the
**real** integration checks.

---

## 9. Problems found and verified fixes

**Problem 1 — `rag_search` crashed when the model sent `top_k` as a string.**
Running the coordinator end to end raised:
`TypeError: '<=' not supported between instances of 'str' and 'int'`.
The LLM emitted `top_k="5"`.
*Fix*: coerce with `int(top_k)` and return a friendly error for non-numeric values.
*Verified*: the coordinator answers the `upsert` question successfully, and
`test_rag_search_accepts_string_top_k` guards it.

**Problem 2 — internal `settings` argument leaked into the tools.**
ADK exposes every tool parameter to the model as a function argument, so the model was
prompted to fill in `settings` and passed a string:
`AttributeError: 'str' object has no attribute 'target_repo_path'`.
*Fix*: the five public tools now take only user-facing arguments and read configuration
through a module-level `get_settings` seam. `test_required_tool_signatures_are_minimal`
locks the signatures.

**Problem 3 — workers delegated to each other.**
A question produced `coordinator -> onboarding_guide -> code_explainer`, and the final
answer became a confused meta-response.
*Fix*: set `disallow_transfer_to_parent` and `disallow_transfer_to_peers` on every worker
so routing is always one hop from the coordinator.

---

## 10. Sample questions, answers and responsible agents

**Q1 — "Where is the upsert to the Pinecone vector database implemented?"**
→ answered by **`code_searcher`** (via `rag_search`):
> The upsert to the Pinecone vector database is implemented in `vector_store.py`, in the
> `upsert_to_vector_db()` function at lines 81-171. It upserts the vectors and metadata
> into the Pinecone index and falls back to a local JSON store when no Pinecone key is
> configured.

Source: `vector_store.py` lines 81-171 (similarity 0.7603).

**Q2 — "What does the file query_pipeline.py do?"**
→ answered by **`code_explainer`** (read the whole file via MCP):
> `query_pipeline.py` is the online query pipeline (stages 5-8). It embeds the user's
> question, retrieves the top-k chunks, injects them into a grounded prompt, and asks the
> LLM for an answer citing only the retrieved context. Its functions are `embed_query`,
> `retrieve_top_k`, `build_prompt_with_context`, `generate_answer` and `run_query_pipeline`.

Source: `query_pipeline.py` (read in full).

**Q3 — "How do the modules in this project import and depend on each other?"**
→ answered by **`architecture_mapper`** (deterministic `ast` analysis):
> `config.py` imports `pathlib` and `dotenv`. `indexer.py` imports `config`, `vector_store`
> and `openai`. `main.py` imports `config`, `indexer` and `query_pipeline`.
> `query_pipeline.py` imports `config` and `vector_store`. `vector_store.py` imports
> `config` and `pinecone`. The CLI entry point is `main.py` (it has a `__main__` guard).

Source: `inspect_imports` output per file.

---

## 11. Security notes

- Credentials live only in `.env` (git-ignored). `.env.example` contains placeholders.
- `describe_configuration()` and `describe_mcp_configuration()` return redacted views for
  diagnostics and never include secret values.
- Missing keys raise a clear `ConfigError` **before** any network call, and the offline
  test suite runs with no keys at all.
- The MCP server is confined to the target project and receives a sanitised environment.
- The vector store never deletes the index or vectors; deletion requires a deliberate
  action outside this codebase.

---

## 12. Known limitations

- Chunk quality is limited by line-based splitting (§4).
- The 11B chat model can still add a claim a tool did not report (e.g. suggesting an
  import that `inspect_imports` did not list); the agent instructions tell it not to, but
  model behaviour is not fully deterministic. Treat tool output as the ground truth.
- `adk web` requires a browser; it is the same agent package as `adk run`.
