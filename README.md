# Multi-Agent Code Helper
An AI assistant that helps developers understand an unfamiliar Python codebase. Ask where something is implemented, what a file does, how modules connect, or how to get started—and receive answers grounded in the target repository.
Built with **Google ADK**, one coordinator, four specialist workers, a custom RAG pipeline, **NVIDIA NIM** embeddings, and **Pinecone** serverless storage. It supports ADK terminal and web interfaces; it does not include a custom website.
## 1. Architecture
```text
User
 |
Coordinator (deterministic fast path + routes other requests)
 |-- code_searcher       -> rag_search -> RAG / Pinecone
 |-- code_explainer      -> Filesystem MCP + summarize_project_file / read_project_file
 |-- architecture_mapper -> list_project_files + inspect_imports (Python AST)
 `-- onboarding_guide    -> get_project_overview (files + README)
```
Editable diagram: [`docs/architecture.drawio`](docs/architecture.drawio), open with [diagrams.net](https://app.diagrams.net).
The coordinator has no tools of its own and delegates to one worker for open-ended questions. A narrow deterministic fast path answers six shapes before any model call: a literal filename lookup ("where is `main.py`?"), a whole-file outline request ("what does `query_pipeline.py` do?"), an import or connection question about named files ("what does `main.py` import?", "how does `main.py` connect to `query_pipeline.py`?"), an obvious "where is/are ...?" question about an operation ("where is the document loaded?", "where are document chunks embedded?"), a project overview or run question ("tell me about this codebase in 5 lines", "how do I run it?"), and a question about the helper's own coordinator/workers. The import answers come from a real `ast` import graph, and the operation answers from real function/class definitions and docstrings, not a model guess. The scope question explains that only `target_project/` is indexed and points at `code_helper/agents/coordinator.py`. These questions therefore still work when the hosted chat endpoint is slow, and no private routing notes reach the user. Workers cannot transfer tasks to peers or back to the coordinator, keeping routing to one coordinator-to-worker hop.
## 2. Technology Stack
| Component | Technology / configuration |
|---|---|
| Language | Python 3.13.7 (Python 3.11+ required) |
| Agent framework | Google ADK 2.12.0 |
| LLM gateway | LiteLLM 1.105.1 |
| Chat model | `openai/gpt-oss-20b` via NVIDIA NIM, with automatic failover to `nvidia/nemotron-3.5-lightning-30b-a3b` and `nvidia/nemotron-3.5-bolt-e4b` |
| Embedding model | `nvidia/nemotron-3-embed-1b` via NVIDIA NIM |
| Vector database | Pinecone serverless, 2048 dimensions, cosine similarity |
| Filesystem tools | Node.js / npx + `@modelcontextprotocol/server-filesystem` |
| Environment | `python-dotenv` |
| Tests | `pytest` 9.1.1 |
The chat model is created once in `code_helper/core/config.py` and shared by all five agents. Embedding configuration is separate from chat configuration.
## 3. Project Layout
```text
demo/
├── code_helper/
│   ├── __init__.py             # Package import entry point
│   ├── agent.py                # ADK web/run root_agent entry point
│   ├── web.py                  # Dev UI launcher with SSE keep-alive
│   ├── core/
│   │   ├── config.py           # Environment, settings and shared chat model
│   │   └── paths.py            # Path-traversal protection
│   ├── agents/
│   │   ├── callbacks.py        # Response hygiene (thoughts and planning text)
│   │   ├── fast_path.py        # Deterministic filename / outline / import / operation answers
│   │   ├── coordinator.py      # Coordinator ("Boss")
│   │   └── workers.py          # Four specialist worker agents
│   ├── integrations/
│   │   └── mcp_filesystem.py   # Restricted, read-only Filesystem MCP
│   ├── rag/
│   │   ├── ingestion.py        # RAG: safe file discovery and reading
│   │   ├── chunking.py         # Line-based chunking
│   │   ├── embeddings.py       # NVIDIA NIM embeddings
│   │   ├── vector_store.py     # Pinecone storage and maintenance
│   │   ├── retrieval.py        # Similarity retrieval and formatting
│   │   ├── reranker.py         # Optional cross-encoder reranking
│   │   └── pipeline.py         # RAG orchestration
│   ├── tools/                  # Agent tool implementations
│   └── index_codebase.py       # Indexing CLI (including --prune)
├── target_project/              # Repository being analysed
├── tests/                       # Offline pytest suite
├── scripts/                     # Live integration checks
├── docs/                        # Architecture diagram and planning
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Test and lint dependencies
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
| 1. Ingestion | `rag/ingestion.py` | Finds files; skips Git, virtual environments, caches, secrets, binaries, and oversized files |
| 2. Chunking | `rag/chunking.py` | Splits code into overlapping line-based chunks |
| 3. Embeddings | `rag/embeddings.py` | Creates NVIDIA NIM embeddings with batching, retries, and dimension checks |
| 4. Storage | `rag/vector_store.py` | Stores vectors and file/line/text metadata in Pinecone |
| 5. Retrieval | `rag/retrieval.py` | Embeds the query with the same model and returns top-k results with source details |
| 6. Orchestration | `rag/pipeline.py` | Coordinates indexing and answer generation |
A literally named file (for example, "where is `main.py`?") is read directly rather than searched semantically, so it is always returned when it exists; a missing file is reported explicitly. Named-file output is capped to the first few chunks so a large file cannot flood the prompt; `summarize_project_file` and `read_project_file` are used for deeper inspection.
Default chunk size is **100 lines**, with **20 lines of overlap**. Configure these using `CHUNK_SIZE` and `CHUNK_OVERLAP`. Each chunk records its file, line range, stable ID, and original text. Each chunk is embedded with its relative path prepended (`# file: <path>`) so filename and module queries match the right file; re-run the indexer after changing this contract.
**Chunking limitations:** chunks may begin mid-function or split long functions; overlap only partly mitigates this. Whitespace-heavy or repetitive files may produce low-information or similar chunks. Overlap also increases embedding cost and index size.
## 5. Agents and Tools
| Agent | Responsibility | Main tool(s) |
|---|---|---|
| `coordinator` | Routes a question to one worker | Delegation only |
| `code_searcher` | Finds where code is implemented | `rag_search` (up to 3 refined queries) |
| `code_explainer` | Explains a file's purpose and contents | Filesystem MCP, `summarize_project_file`, `read_project_file` |
| `architecture_mapper` | Lists files and analyses imports | `list_project_files`, `inspect_imports` |
| `onboarding_guide` | Explains purpose, setup, and reading order | `get_project_overview` |
The five required custom Python tools are `rag_search`, `read_project_file`, `list_project_files`, `inspect_imports`, and `get_project_overview`; `summarize_project_file` is an additional deterministic outline helper. The deterministic tools do not use AI — for example, `inspect_imports` and `summarize_project_file` parse code with Python's `ast` module. The coordinator's fast path also builds an import graph (`code_helper/tools/import_analysis.py`) to answer "what does `main.py` import?" and two-file connection questions without a model. Internal settings are loaded by the tools rather than exposed as model-supplied arguments. Workers also enforce per-agent tool and model-call budgets (`MAX_TOOL_CALLS`, `MAX_LLM_CALLS`) so a run cannot loop indefinitely; `code_searcher` may refine its query up to the `MAX_TOOL_CALLS` budget (default 3) instead of giving up after one search.
## 6. Filesystem MCP and Security
The `code_explainer` uses ADK's `McpToolset` and `StdioConnectionParams` to connect to the official Filesystem MCP server.
- MCP is restricted to the `target_project/` directory.
- Only allow-listed read tools are exposed; write, edit, move, and create operations are not exposed.
- The subprocess receives a sanitised environment that removes variables containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, or `CREDENTIAL`.
- If MCP is unavailable, the deterministic `summarize_project_file` and `read_project_file` tools remain as fallbacks. Set `DISABLE_MCP_FILESYSTEM=1` to disable MCP deliberately.
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
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # adds pytest and ruff
```
### Configure environment
Run from `demo/`:
```powershell
Copy-Item .env.example .env
```
Add the required values to `.env`: `NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`), `NVIDIA_BASE_URL`, `NVIDIA_CHAT_MODEL`, `NVIDIA_EMBED_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and `PINECONE_REGION`.
Optional settings include `TARGET_REPO_PATH`, `EMBEDDING_DIMENSION`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `PINECONE_NAMESPACE`, `MAX_LLM_CALLS`, `MAX_TOOL_CALLS`, `NVIDIA_MAX_TOKENS`, `NVIDIA_TIMEOUT_SECONDS`, `NVIDIA_FALLBACK_CHAT_MODELS` (comma-separated; `off` disables failover), `NVIDIA_FALLBACK_TIMEOUT_SECONDS`, `NVIDIA_PRIMARY_COOLDOWN_SECONDS`, `NVIDIA_REASONING_EFFORT` (default `low`; empty disables the hint), and `NVIDIA_RERANK_MODEL` (leave empty to disable reranking).
### Index and verify the target project
```bash
python -m code_helper.index_codebase --progress
python -m code_helper.index_codebase --prune --progress
python -m code_helper.index_codebase --stats
```
`--prune` deletes vectors in the dedicated namespace that are not part of the current chunk set. It is opt-in: normal indexing never deletes data.
### Run the agents
Run these commands from `demo/`, the directory containing `code_helper/`:
```bash
adk run code_helper
python -m code_helper.web --port 8000
```
`adk run` provides a terminal interface. `python -m code_helper.web` serves the same ADK web UI as `adk web`, with two fixes for long model calls: a keep-alive comment every 10 seconds on `/run_sse`, so the browser does not abort a silent stream with `TypeError: Failed to fetch`, and UTF-8 console output for LiteLLM's emoji logs. Open `http://127.0.0.1:8000/dev-ui/` and select `code_helper`.
The launcher keeps sessions in memory by default; pass `--session-service-uri sqlite:///./.adk/sessions.db` to persist them across restarts. `--heartbeat-seconds 0` disables the keep-alive.
Chat responses contain only the final answer: reasoning (`thought=True`) parts are dropped at the shared model layer before ADK renders them, and the callback sanitizer removes any remaining planning line.
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
- **Offline tests:** `python -m pytest` - **209 passed**. Tests are mocked and do not require keys or network access.
- **NVIDIA NIM embeddings:** embedding model verified; vector dimension **2048**.
- **Pinecone:** 6 chunks indexed after `--prune`; index dimension 2048 with cosine similarity. A test query about vector upsert ranked `vector_store.py` first.
- **Filesystem MCP:** server exposed 14 tools; all 10 allow-listed read-only tools were present, and a project file was read through MCP.
- **Agent routing:** the coordinator routed test questions to `code_searcher`, `code_explainer`, and `architecture_mapper`.
- **ADK terminal:** a query about the upsert implementation returned `vector_store.py` with source line ranges.
- **ADK web:** server started locally and `/list-apps` returned HTTP 200 for `code_helper`.

### Measured latency
- **Retrieval (warm):** ~0.5–0.9 s per question (query embedding + Pinecone query).
- **First question per process:** ~4–5 s once, while the Pinecone client resolves the index host; the handle is reused afterwards.
- **Model calls dominate:** routing plus a tool-using worker means 2–3 sequential LLM calls. `NVIDIA_MAX_TOKENS` (default 800) bounds generation length; lower it for faster, shorter answers. `NVIDIA_TIMEOUT_SECONDS` (default 60) bounds each primary attempt at the ADK layer so provider adapters cannot override it. If the primary times out or errors, the shared model retries the same request on the configured fallback models; `NVIDIA_PRIMARY_COOLDOWN_SECONDS` (default 300) then skips the dead primary so later steps do not pay the timeout again.
- **Reasoning budget:** `NVIDIA_REASONING_EFFORT=low` is sent through `extra_body` (the NVIDIA NIM provider rejects it as a top-level LiteLLM argument). This reduced a simple model call from 5.7 s to 1.2 s. `thought=True` reasoning parts are dropped before the Dev UI renders them, and a worker callback removes any leading planning line that still slips into an answer.
- **Live end-to-end:** two model-backed questions (a located file and an absent topic) completed in ~39 s total, with correct answers and no timeout; a single question after a cold start was ~31–34 s. A file explanation using the AST outline completed in ~44 s. With the endpoint degraded, two deterministic fast-path questions completed in ~29 s total without a model call.
- **Live failover run (degraded primary):** the primary `openai/gpt-oss-20b` endpoint stopped responding, so the first model-backed question paid the 60 s bound and then answered from `nvidia/nemotron-3.5-lightning-30b-a3b`; later model calls skipped the dead primary via the breaker. Final eight-question run: `Where is the document loaded?` (0.3 s, deterministic AST locator), `How does main.py connect to query_pipeline.py?` (0.1 s, deterministic import graph), `Where are document chunks embedded?` (0.1 s, deterministic AST locator), `Where is the local JSON fallback used?` (0.1 s, deterministic AST locator), `How does the query pipeline retrieve context?` (78.7 s, first model call after the primary timeout), `What files does main.py import?` (0.1 s, deterministic import graph), `What is the entry point of this project?` (0.1 s, deterministic overview), and `How does the indexing pipeline store vectors?` (19.1 s, model with breaker open). No answer was the timeout notice, a clarification request, or an invented file.
- **Web UI after the fixes:** sequential questions in the Dev UI (`Where is the Pinecone upsert implemented?`, `How is the embedding retry logic implemented?`, `Where is main.py?`) completed in 12.1 s, 16.2 s, and 3.0 s. Answers contained no `THOUGHT` blocks, the browser reported no `Failed to fetch`, and a 60 s silent model turn received five keep-alive comments without disconnecting.
- **MCP cold start:** the first explainer call starts the `npx` filesystem server (a few seconds); later calls reuse it.

The scripts above are live integration checks; the offline unit tests are mocked.
## 9. Issues Fixed
1. **String `top_k`:** the model sometimes supplied `"5"` as a string. The tool now converts it to an integer and returns a friendly error for invalid input; a regression test covers this.
2. **Leaked internal settings argument:** ADK exposed an internal `settings` parameter to the model. Public tools now accept only user-facing arguments and obtain configuration through `get_settings`.
3. **Worker-to-worker delegation:** workers could previously delegate to each other and produce confused answers. Transfer restrictions now enforce one-hop coordinator-to-worker routing.
4. **Filename queries could miss the file:** "where is `main.py`?" searched semantically, and the top-5 chunks could omit `main.py` entirely, leaving the model to speculate from imports. `rag_search` now resolves a literally named file directly and reports a missing file explicitly. Chunks are also embedded with their relative path (`# file: <path>`), so module-name queries match the right file.
5. **Absent topics returned confident chunks:** "Where is the Kubernetes deployment configured?" returned unrelated `main.py`/`config.py` chunks and let the model guess. `rag_search` now requires at least one retrieved chunk to mention a meaningful query term; otherwise it reports "No relevant evidence" and the searcher must answer "Not found in the indexed code." Routing also sends semantic "where is ...?" questions to `code_searcher` (literal filenames use the deterministic fast path), and the architecture mapper no longer inspects arbitrary files.
6. **Slow repeated retrieval:** every question built a fresh Pinecone client, which resolved the index host over the control plane (measured 8 s, sometimes 27 s with retries). Retrieval now reuses one Pinecone handle and one embedding client, memoises query embeddings, and caps chat output with `NVIDIA_MAX_TOKENS` (default 800). Warm retrieval is ~0.5–0.9 s.
7. **Unbounded chat requests:** a stalled NVIDIA NIM completion could block a session indefinitely (observed during latency testing). The shared model now enforces `NVIDIA_TIMEOUT_SECONDS` (default 60) with `asyncio.wait_for`, so the bound holds even when LiteLLM ignores its own `timeout`, and disables LiteLLM's hidden retries. If the bound fires, the agent now returns a short retry message instead of ending with an empty response and a traceback.
8. **Retired chat model:** `meta/llama-3.2-11b-vision-instruct` started hanging, while Llama 3.1/3.3 fallbacks returned HTTP 410/404. The default is now `openai/gpt-oss-20b`, verified with live tool calls and routing; `NVIDIA_CHAT_MODEL` remains overridable.
9. **Stale vectors after re-indexing:** changing chunking parameters left old vectors in Pinecone, producing duplicate and outdated line ranges. `--prune` removes vectors that are not in the current chunk set; the demo index is now 6 vectors.
10. **Reranker on the interactive path:** reranking is implemented and fail-open, but the hosted cross-encoder took over 40 s even for five short passages. It is disabled by default (`NVIDIA_RERANK_MODEL` empty); enable it only when the extra latency is acceptable.
11. **Windows console crash:** live scripts now reconfigure stdout/stderr to UTF-8, so model output containing Unicode cannot raise `UnicodeEncodeError`.
12. **Planning text in answers:** `gpt-oss-20b` sometimes prefixed a correct answer with a planning line. `NVIDIA_REASONING_EFFORT=low` reduces it, and `agents/callbacks.py` deterministically strips any leading planning line from final worker/coordinator text without touching tool-call turns.
13. **File-explanation timeout:** sending a whole source file to the chat model could exceed the 60 s model budget and produce no answer. `code_explainer` now calls `summarize_project_file` for a compact AST outline (docstring, imports, constants, classes, functions, line numbers) and reads the full file only when needed. A live explanation completed in ~44 s.
14. **Slow hosted model blocked even routing:** the NVIDIA endpoint can take 20 s for a one-word reply and occasionally exceeds 60 s on the routing call. `agents/fast_path.py` now answers literal filename lookups, whole-file outline requests, project overview/run questions, and helper-scope questions deterministically before any model call; a bounded timeout returns a retry message instead of an empty response. The overview reuses the same evidence-only facts as `get_project_overview`, so it never invents purpose or commands. Under the same degraded endpoint, two fast-path questions completed in ~29 s total.
15. **Planning text and a misleading clarification:** the coordinator could show `"User wants overview. Use onboarding_guide."` as its visible answer, and `code_explainer` asked "which file?" when the subject was the helper's own coordinator. The sanitizer now strips `User wants`, `User needs`, `User requested` and similar prefixes (a planning-only turn becomes empty instead of leaking), and helper-scope questions are answered directly with the indexed scope and the location of `code_helper/agents/coordinator.py`.
16. **Agent thoughts in the web UI:** streaming reasoning arrived as `thought=True` parts, which the ADK Dev UI renders as visible `THOUGHT` blocks. The shared model wrapper now drops thought-only chunks before ADK aggregates them, so the chat shows the final answer without interrupting it with private planning. The `strip_reasoning` callback repeats the filter as a fallback.
17. **Second question failed with `TypeError: Failed to fetch`:** `/run_sse` writes nothing while a model call is pending, and a slow NVIDIA NIM turn can leave the stream silent for over a minute, which can be treated as a dead connection. `code_helper/web.py` wraps the ADK app in an ASGI middleware that writes SSE keep-alive comments during idle periods; it also enables auto-created sessions and tags the launcher's output streams as UTF-8.
18. **NIM streaming tool calls aborted runs:** the streaming parser intermittently rejected gpt-oss tool-call deltas (`MidStreamFallbackError: ... unexpected tokens remaining in message header: Some("to=functions.rag_search")`), leaving the chat with no answer. The shared model now requests the complete completion instead of token deltas; the UI only shows the final answer anyway, so nothing user-visible is lost, and the keep-alive covers the wait.
19. **A dead primary model returned only timeout text:** the `openai/gpt-oss-20b` endpoint sometimes accepts a request and never responds (a raw HTTP probe timed out after 120 s, while LiteLLM reported 183 s despite `num_retries=0`). Seven of eight obvious questions therefore returned the retry notice before the coordinator could even route. The shared model now retries the same request on `nvidia/nemotron-3.5-lightning-30b-a3b` and `nvidia/nemotron-3.5-bolt-e4b`, then opens a circuit breaker for `NVIDIA_PRIMARY_COOLDOWN_SECONDS` so later steps skip the dead primary. Both fallbacks were verified with live NIM tool calls, and the failover path has regression tests.
20. **Obvious import/connection questions needed a model:** "what does `main.py` import?" and "how does `main.py` connect to `query_pipeline.py`?" now use a deterministic `ast` import graph before any model call. One-file questions also answer in both directions ("what imports `alpha.py`?" / "where is `alpha.py` used?"), and two-file questions show a direct or multi-hop import path with line numbers. This is static import evidence only; the answer says so. `code_searcher` may also refine its query up to three times instead of stopping after one weak search.
21. **An obvious "where is ...?" question asked for clarification:** the coordinator sometimes answered `Where is the document loaded?` with "Which document are you referring to?" instead of routing it. The fast path now locates unambiguous operations from real function/class definitions and docstrings (`code_helper/tools/symbol_locator.py`), for both "where is" and "where are" wording. It ranks a definition whose name contains the question's operation above one that merely mentions the words, reports ties within a file together, and leaves cross-file ties to the workers. `Where is the document loaded?` now returns `indexer.py` / `load_document`, and `Where are document chunks embedded?` returns `indexer.py` / `embed_chunks`, in ~0.1 s.
22. **A search hint was not enough to stop a wrong citation:** for `Where are document chunks embedded?`, retrieval ranked `vector_store.py` first and the model cited the storage helper even though the tool said `indexer.py` contained every query term. `rag_search` now also moves the sole full-term match to the first result, so a model that cites the top block lands on the defining file. The snippet window is 4000 characters (was 1200), which keeps a small file's relevant function visible instead of only its opening lines.
## 10. Example Questions
- **“Where is `main.py`?”** the deterministic fast path returns its path and one-sentence purpose without a model call.
- **“What does `query_pipeline.py` do?”** the deterministic fast path returns the AST outline (imports, constants, functions, docstrings); ask "walk me through `query_pipeline.py`" for a prose explanation from `code_explainer`.
- **“What files does `main.py` import?”** / **“How does `main.py` connect to `query_pipeline.py`?”** the deterministic fast path reads the `ast` import graph and returns the actual import statements with line numbers.
- **“Where is the document loaded?”** / **“Where are document chunks embedded?”** the deterministic fast path locates the real function and docstring (`load_document`, `embed_chunks`) and returns the file plus line range without a model call.
- **“Tell me something about this codebase in 5 lines”** the deterministic fast path returns a short evidence-based summary: purpose, files, entry points, dependencies, run commands and reading order.
- **“Explain how the coordinator agent works”** the deterministic fast path explains that only `target_project/` is indexed and points at `code_helper/agents/coordinator.py`; it never asks which file or guesses from general knowledge.
- **“Where is the upsert to Pinecone implemented?”** `code_searcher` uses RAG to locate the relevant function and return file/line references.
- **“How do modules import and depend on one another?”** `architecture_mapper` uses deterministic Python AST analysis to inspect imports and entry points.
## 11. Known Limitations
- Line-based chunks may not preserve complete function boundaries.
- The language model may occasionally add a claim not supported by tool output; treat retrieved code and deterministic tool results as the ground truth.
- Only `TARGET_REPO_PATH` is indexed (by default `target_project/`). Questions about the helper's own source are answered with a scope note and a pointer to `code_helper/agents/coordinator.py`; to search the helper itself, set `TARGET_REPO_PATH=./` and re-index.
- `adk web` requires a browser and uses the same agent package as `adk run`.
