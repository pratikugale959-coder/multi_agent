# Planning Document — Multi-Agent Code Helper

> **Purpose.** This is the implementation plan for the Multi-Agent Code Helper,
> written before the code was built. It defines the problem, requirements,
> proposed architecture, module contracts, delivery phases, risks and acceptance
> criteria that the implementation must satisfy.
>
> **Status.** Design baseline v1.0 — approved for implementation.
>
> **Companion documents.** [`README.md`](../README.md) for setup and usage, and
> [`architecture.drawio`](architecture.drawio) for the editable architecture map.

- **Project:** Multi-Agent Code Helper
- **Planned:** 2026-10-16
- **Deliverable:** the `demo/` project (Python package `code_helper/`)
- **Target project:** a separate, unfamiliar Python repository in `target_project/`

---

## 1. Executive summary

We will build an AI assistant that helps a developer understand an unfamiliar
Python repository. It answers four recurring questions — *where* code is, *what*
a file does, *how* modules connect, and *how* to get started — using Google ADK,
NVIDIA NIM models and Pinecone.

The design is a coordinator agent with **no tools of its own** that delegates to
one of four specialist workers. Every worker answer is grounded in a tool result,
and the RAG pipeline is written by hand (no LangChain or LlamaIndex) so each stage
is explicit and testable. A read-only Filesystem MCP integration provides safe
access to the target project.

---

## 2. Problem statement

A developer joining an unfamiliar Python codebase repeatedly asks:

1. *Where* is a given feature, function or variable implemented?
2. *What* does a particular file do?
3. *How* do the modules import and depend on one another?
4. *How* do I install, run and navigate the project?

Generic chatbots answer from memory and invent filenames, functions and APIs. The
project must instead answer **only from the target repository**, cite the file and
line range, and be verifiable by the reader.

**Problem statement:** build a multi-agent assistant that answers the four
questions above from a real codebase, with every claim traceable to evidence,
using Google ADK, NVIDIA NIM and Pinecone, and without a RAG framework.

---

## 3. Goals and non-goals

### 3.1 Goals

| # | Goal |
|---|---|
| G1 | Route each of the four question types to the most appropriate specialist worker. |
| G2 | Ground every answer in a real file and line range, or state that no evidence exists. |
| G3 | Implement the RAG pipeline by hand as separate, testable stages. |
| G4 | Prefer deterministic analysis (`ast`, file reads) over model guesswork. |
| G5 | Share a single chat-model configuration across all five agents. |
| G6 | Keep the whole test suite runnable offline, with no keys and no network. |
| G7 | Prevent runaway agent loops and unbounded cost with explicit call budgets. |
| G8 | Make file access read-only and confined to the target project. |

### 3.2 Non-goals

- A custom website — the ADK terminal (`adk run`) and web (`adk web`) interfaces are used.
- Supporting languages other than Python.
- LLM-based chunking, graph databases or framework-managed RAG.
- Destructive vector-store operations — the code never deletes the index.
- Multi-hop agent conversations — routing is a single coordinator-to-worker hop.

---

## 4. Users and use cases

| # | User | Use case |
|---|---|---|
| U1 | New developer | "Where is X implemented?" |
| U2 | Reviewer | "What does this file do?" |
| U3 | Tech lead | "How are the modules connected?" |
| U4 | Onboarding engineer | "How do I run this and where do I start?" |

**Primary flow:** the user asks a question; the coordinator classifies it and
delegates to one worker; the worker calls one or more tools; the worker returns a
cited answer; the coordinator returns that answer unchanged.

---

## 5. Requirements

### 5.1 Functional requirements

| # | Requirement |
|---|---|
| FR1 | The coordinator must select exactly one worker for each request. |
| FR2 | `code_searcher` must locate where a feature or file is implemented. |
| FR3 | `code_explainer` must explain a complete source file. |
| FR4 | `architecture_mapper` must report real imports, definitions and entry points. |
| FR5 | `onboarding_guide` must explain purpose, setup and reading order. |
| FR6 | Semantic search must return file, line range and similarity score. |
| FR7 | A query that names a file (e.g. `main.py`) must resolve that file directly. |
| FR8 | A missing file or empty result must be reported as "not found", never invented. |
| FR9 | The project must be indexable from the command line. |
| FR10 | The explainer must read files through MCP, with a local fallback. |

### 5.2 Non-functional requirements

| # | Requirement |
|---|---|
| NFR1 | Deterministic core: the same input yields the same analysis. |
| NFR2 | Offline testability: no keys or network required for the unit suite. |
| NFR3 | Security: read-only file access, no secret leakage, path-traversal protection. |
| NFR4 | Clear failures: missing credentials raise before any network call. |
| NFR5 | Reproducibility: pinned model ids and typed, validated configuration. |
| NFR6 | Readability: one module per RAG stage and documented public functions. |
| NFR7 | Bounded execution: tool and model call budgets per request. |
| NFR8 | Portability: Python 3.11+, Windows and POSIX. |
| NFR9 | Latency: reuse network clients; a warm retrieval should stay under about one second, with generation length bounded. |

### 5.3 Constraints

- Python 3.11 or newer (development environment uses 3.13).
- Google ADK with LiteLLM as the model gateway.
- NVIDIA NIM chat and embedding endpoints.
- Pinecone serverless for vector storage.
- Node.js and `npx` available for the Filesystem MCP server.
- No LangChain, LlamaIndex or other RAG framework.

---

## 6. Proposed architecture

### 6.1 Layer map

```text
Clients and entry points   adk run · adk web · scripts/
            |
Agent layer (ADK)          coordinator -> one of four workers
            |
Tool layer                 rag_search · read_project_file / Filesystem MCP
                           list_project_files · inspect_imports · get_project_overview
            |
RAG pipeline               ingestion -> chunking -> embeddings
                           -> vector_store -> retrieval -> rag_pipeline
            |
External services          NVIDIA NIM (chat + embeddings) · Pinecone · MCP server
```

Cross-cutting modules: `config.py` (typed settings), `paths.py` (path guard),
`mcp_filesystem.py` (read-only allow-list) and the shared chat model.

### 6.2 Request lifecycle

1. ADK loads `code_helper` and discovers `root_agent` from `__init__.py`.
2. `setup_invocation_limits` applies the run-level model-call cap.
3. The coordinator chooses one worker and rewrites the request as a task.
4. The worker calls its tool(s); limits are enforced before each tool and model call.
5. The tool returns evidence with file paths and line ranges.
6. Once the tool budget is spent, tools are removed from the next model request so
   the worker must answer in prose.
7. The coordinator returns the worker's answer unchanged, with citations intact.
8. The worker's `after_agent_callback` clears per-invocation counters.

---

## 7. Detailed design

### 7.1 Module plan

| Module | Responsibility |
|---|---|
| `code_helper/config.py` | Load `.env`, build immutable `Settings`, share one chat model. |
| `code_helper/paths.py` | Resolve paths inside the project root; reject traversal. |
| `code_helper/ingestion.py` | Discover and read source files; skip caches, secrets, binaries, oversized files. |
| `code_helper/chunking.py` | Line-window chunking, stable ids, vector metadata, embedding text. |
| `code_helper/embeddings.py` | NVIDIA NIM embeddings with batching, retries and dimension checks. |
| `code_helper/vector_store.py` | Pinecone serverless index; upsert and query; never delete. |
| `code_helper/retrieval.py` | Embed the query, retrieve top-k, build `RetrievedChunk` objects, format output. |
| `code_helper/rag_pipeline.py` | Orchestrate indexing and answer generation; `IndexReport`. |
| `code_helper/index_codebase.py` | CLI: `--root`, `--progress`, `--stats`. |
| `code_helper/mcp_filesystem.py` | Read-only Filesystem MCP toolset with a sanitised child environment. |
| `code_helper/agent.py` | Coordinator (`root_agent`), routing instruction, run limits. |
| `code_helper/agents.py` | Four workers, their instructions, per-agent call limits. |
| `code_helper/tools/*.py` | The five custom tools exposed to the workers. |

### 7.2 Agent graph

| Agent | Type | Responsibility | Tool(s) | AI used? |
|---|---|---|---|---|
| `coordinator` | root | Route to one worker; return its answer | none | routing only |
| `code_searcher` | worker | Where is X implemented? | `rag_search` | yes |
| `code_explainer` | worker | What does a whole file do? | Filesystem MCP + `read_project_file` | optional |
| `architecture_mapper` | worker | Imports, dependencies, entry points | `list_project_files`, `inspect_imports` | no |
| `onboarding_guide` | worker | Purpose, setup, reading order | `get_project_overview` | no |

Rules:

- The coordinator has no tools, so it cannot answer directly.
- Workers cannot transfer to peers or back to the coordinator.
- Every worker has at least one tool; at least one tool uses no AI.

### 7.3 Tool contracts

| Tool | Signature | Deterministic? | Source of truth |
|---|---|---|---|
| `rag_search` | `(query: str, top_k: int = 5) -> str` | mixed | Pinecone + exact file reads |
| `read_project_file` | `(file_path: str) -> str` | yes | target filesystem |
| `list_project_files` | `() -> str` | yes | target filesystem |
| `inspect_imports` | `(file_path: str) -> str` | yes | Python `ast` |
| `get_project_overview` | `() -> str` | yes | files + README |

Public signatures expose **only user-facing arguments**. Configuration is resolved
inside each tool through `get_settings`, so ADK never asks the model to invent
internal arguments.

`rag_search` has two paths:

1. **Named file** — if the query contains a literal filename (for example
   `main.py`), the file is read through `read_source_file`, chunked and returned
   in full with a similarity of `1.0`. Semantic search is skipped. If the file is
   missing, excluded or outside the root, the tool returns a definitive
   "not found" message and does not fall back to loose matches.
2. **Descriptive query** — the query is embedded and matched against Pinecone;
   the top-k chunks are returned with file, line range and score. Before returning
   them, the tool checks that at least one chunk mentions a meaningful query term
   (stopwords excluded). If none does — for example a question about Kubernetes or
   another topic the project does not contain — it reports "No relevant evidence"
   instead of offering unrelated code.

### 7.4 RAG pipeline

| Stage | Module | Design |
|---|---|---|
| 1. Ingestion | `ingestion.py` | Walk the target; skip `.git`, virtual environments, caches, secrets, binaries and files over 1 MB. |
| 2. Chunking | `chunking.py` | Line windows of 100 lines with 20 lines of overlap; stable ids `file:start-end`. |
| 3. Embeddings | `embeddings.py` | `nvidia/nemotron-3-embed-1b`, batched, retried, dimension-validated (2048). |
| 4. Storage | `vector_store.py` | Pinecone serverless, cosine, deterministic vector ids, metadata for provenance. |
| 5. Retrieval | `retrieval.py` | Embed the query with the same model; return top-k with scores and formatting. |
| 6. Orchestration | `rag_pipeline.py` | `index_repository()` (stages 1–4) and `answer_question()`; `IndexReport` summary. |

**Path-aware embedding.** Each chunk is embedded as:

```text
# file: <relative_path>
<chunk.text>
```

The path is prepended so a query naming a file or module is drawn to that file's
own chunks, not only to chunks whose body happens to contain the word. The stored
chunk `text` and displayed snippets stay path-free. Existing indexes must be
rebuilt after changing this contract.

**Chunking trade-off.** A chunk may begin mid-function, so retrieved text is a
*pointer*, not a self-contained unit. Overlap mitigates but does not remove this.

### 7.5 Data contracts

**`Chunk`** (frozen dataclass): `chunk_id`, `file_path`, `line_start`,
`line_end`, `text`, `metadata`.

- `to_vector_metadata()` returns only scalars: `chunk_id`, `file_path`,
  `line_start`, `line_end`, `text` (truncated to 40,000 characters).
- `embedding_text` returns `# file: <file_path>\n<text>`.

**`RetrievedChunk`** (frozen dataclass): `chunk_id`, `file_path`, `line_start`,
`line_end`, `score`, `text`; `line_range` renders `start-end`.

**`IndexReport`**: `root`, `files_discovered`, `files_skipped`, `chunks_created`,
`chunks_indexed`, `embedding_dimension`, `index_name`, `errors`.

**Pinecone metadata keys:** `chunk_id`, `file_path`, `line_start`, `line_end`,
`text`.

### 7.6 Configuration

`Settings` is an immutable snapshot built once from the environment and cached.
Embedding configuration is kept separate from chat configuration.

| Variable | Required | Default / notes |
|---|---|---|
| `NVIDIA_API_KEY` or `NVIDIA_NIM_API_KEY` | yes | Either name is accepted and kept in sync. |
| `PINECONE_API_KEY` | yes | Required before any Pinecone call. |
| `NVIDIA_BASE_URL` | no | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_CHAT_MODEL` | no | `meta/llama-3.2-11b-vision-instruct` |
| `NVIDIA_EMBED_MODEL` | no | `nvidia/nemotron-3-embed-1b` |
| `PINECONE_INDEX_NAME` | no | `code-helper` |
| `PINECONE_CLOUD` | no | `aws` |
| `PINECONE_REGION` | no | `us-east-1` |
| `PINECONE_NAMESPACE` | no | empty |
| `TARGET_REPO_PATH` | no | `target_project/` inside the project |
| `EMBEDDING_DIMENSION` | no | `2048` |
| `CHUNK_SIZE` | no | `100` lines |
| `CHUNK_OVERLAP` | no | `20` lines |
| `TOP_K` | no | `5` |
| `MAX_LLM_CALLS` | no | `10` |
| `MAX_TOOL_CALLS` | no | `3` |
| `NVIDIA_MAX_TOKENS` | no | `800`; caps chat output length |
| `NVIDIA_TIMEOUT_SECONDS` | no | `60`; bounds a stalled chat request |
| `DISABLE_MCP_FILESYSTEM` | no | `0`; set to `1` to skip MCP and use the local reader |

### 7.7 Run limits and loop protection

| Control | Design |
|---|---|
| Run-level model cap | `MAX_LLM_CALLS` is applied with `min(existing RunConfig limit, configured limit)`. |
| Per-agent tool budget | Searcher and onboarding guide: 1; explainer: 2; architecture mapper: `MAX_TOOL_CALLS` (3). |
| Duplicate tool call | An identical `(agent, tool, arguments)` call in one invocation is rejected with an instruction to answer. |
| Budget exhausted | Tools are removed from the next model request, forcing a textual answer. |
| Worker model cap | A worker may make at most 5 model calls per request. |
| State cleanup | Per-invocation counters are cleared by each worker's `after_agent_callback`. |

### 7.8 Error handling

- Invalid user input is returned as a short, actionable string; tools do not raise
  for bad arguments.
- Missing credentials raise `ConfigError` before a network call.
- Embedding and vector-store failures raise typed errors and are converted to
  readable tool messages.
- A file read failure (missing, excluded, binary, too large, outside the root) is
  reported, never guessed.

---

## 8. Key design decisions

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| D1 | Coordinator plus four workers; workers cannot transfer | Routing stays one hop and answers stay attributable | Free-form hand-off caused confused meta-answers |
| D2 | Coordinator has no tools | Forces every answer through a worker's evidence | A tool-using coordinator blurred responsibility |
| D3 | Custom RAG, one module per stage | Makes each stage explicit and unit-testable | Frameworks hid the stages and added dependencies |
| D4 | Deterministic `ast` tools | Imports and definitions are facts | Asking the model to infer imports was unreliable |
| D5 | One shared chat model | Explicit requirement; one configuration point | Per-agent models risked inconsistent behaviour |
| D6 | Separate embedding configuration | Prevents swapping chat and embedding models | One shared "model" setting was error-prone |
| D7 | Typed `Settings` with fail-fast credential guards | Single validated source of truth | Scattered `os.getenv` calls |
| D8 | Line-based chunking | Simple, deterministic, dependency-free | Syntax-aware chunking added scope |
| D9 | Path-aware embedding text | Filename queries match their own file | Embedding text only lost filename signal |
| D10 | Exact file resolution before semantic search | A small index can omit the named file from top-k | Relying on similarity alone |
| D11 | Read-only Filesystem MCP | Real MCP integration without write access or secrets | Full MCP toolset was unsafe |
| D12 | Per-agent tool and model budgets | Prevents loops and bounds latency and cost | Unbounded agent loops |
| D13 | Lexical relevance gate on semantic results | A similarity score alone cannot prove relevance; absent topics must be refused | Returning top-k unconditionally invited speculation |
| D14 | Reuse network clients and memoise query embeddings | A fresh Pinecone handle resolves the index host per call (seconds); clients are stateless | Constructing a client per query |
| D15 | Bound chat output and request time | Latency grows with output tokens; a stalled call can hang a session, and provider timeouts are not reliable | Unbounded generation and default retries |

---

## 9. Security plan

| Concern | Control |
|---|---|
| Secrets at rest | Keys live in `.env` (git-ignored); `.env.example` contains placeholders. |
| Secrets in diagnostics | `describe_configuration()` returns redacted booleans, never values. |
| Missing keys | Fail fast with `ConfigError` before any network call. |
| Path traversal | `resolve_in_project` confines every path to the target root. |
| File access | MCP exposes only read tools; write, edit, move and create are never surfaced. |
| Child process | `npx` receives a sanitised environment with `KEY`, `TOKEN`, `SECRET`, `PASSWORD` and `CREDENTIAL` variables removed. |
| Secret files | Ingestion refuses `.env`, `*.key`, `*.pem` and similar files. |
| Vector data | The code never deletes the index or vectors. |

---

## 10. Testing strategy

- **Offline first.** External services are replaced by fakes, so the suite needs no
  keys and no network.
- **One test module per stage:** ingestion, chunking, embeddings, vector store,
  retrieval, pipeline, agents, tools, config and MCP.
- **Determinism guarded by tests:** chunk boundaries and overlap, frozen
  dataclasses, tool signatures, read-only MCP filtering, secret stripping and the
  path-aware `embedding_text`.
- **Regression tests for known traps:** string `top_k`, leaked `settings`
  arguments, worker-to-worker transfers, `MAX_LLM_CALLS` capping, counter cleanup
  and filename resolution.
- **Live integration is separate.** `scripts/` contains optional checks for
  embeddings, indexing/search, MCP tool listing and agent routing; these require
  keys and network and are not part of pytest.

Planned commands:

```bash
python -m pytest
ruff check .
ruff format --check .
```

---

## 11. Implementation plan

| Phase | Deliverable | Exit criteria |
|---|---|---|
| P0 | Reconnaissance: confirm models, providers, dimensions | Model ids and the 2048 dimension are confirmed. |
| P1 | Configuration and path guard | `Settings` builds from `.env`; traversal is rejected. |
| P2 | Ingestion and chunking | Files are discovered safely; chunks have stable ids and line ranges. |
| P3 | NVIDIA NIM embeddings | Batched embeddings validate the configured dimension. |
| P4 | Pinecone vector store | Index creation, upsert and query work; no delete path exists. |
| P5 | Retrieval and formatting | Top-k results carry file, line range and score. |
| P6 | Pipeline orchestration and CLI | `index_repository` and `index_codebase` run end to end. |
| P7 | Agents | Coordinator plus four workers construct and share one model. |
| P8 | Tools | Five tools exist; the deterministic ones use no LLM. |
| P9 | Filesystem MCP | Read-only toolset works with a confined root and sanitised env. |
| P10 | Offline test suite | The full suite passes without keys or network. |
| P11 | Live integration checks | Embeddings, Pinecone, MCP and routing verified against real services. |
| P12 | Documentation and diagram | README, `docs/PLANNING.md` and `docs/architecture.drawio` are complete. |

Build order is bottom-up: infrastructure and the RAG stages first, agents and tools
on top, then verification and documentation.

---

## 12. Risks and mitigations

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Model sends an argument of the wrong type (e.g. `top_k="5"`). | Medium | Coerce and validate tool inputs; regression test. |
| R2 | Internal configuration leaks into a tool signature. | Medium | Public tools take only user-facing arguments; config is resolved internally. |
| R3 | Embedding dimension differs from the index. | Medium | Validate the returned dimension before upsert. |
| R4 | Workers delegate to each other and confuse the answer. | Medium | Disallow transfers to parent and peers. |
| R5 | A filename query misses the file in top-k. | Medium | Resolve literally named files directly; report missing files explicitly. |
| R6 | The model speculates when evidence is absent. | Medium | Strict searcher instruction; "not found" is an allowed answer. |
| R7 | An agent loops on tool calls. | Medium | Per-agent budgets, duplicate-call rejection and a model-call cap. |
| R8 | MCP exposes write access or inherits secrets. | Low | Read-only allow-list, confined root and a sanitised environment. |
| R9 | Line-based chunks split functions. | Medium | Documented limitation; overlap mitigates. |
| R10 | Index becomes stale after an embedding-contract change. | Medium | Re-run the indexer after changing `embedding_text`. |

---

## 13. Acceptance criteria

- [ ] The coordinator routes all four question types to the correct worker.
- [ ] `code_searcher` resolves a named file directly and reports a missing file.
- [ ] A semantic query without a filename ranks the expected file first.
- [ ] Answers cite a real file and line range, or state "not found".
- [ ] The RAG pipeline is implemented by hand as separate, documented stages.
- [ ] Embedding text includes `# file: <path>` and the index has been rebuilt.
- [ ] At least one tool works with no AI; four of the five are deterministic.
- [ ] All five agents share one cached chat model.
- [ ] Tool and model budgets bound every request; counters are cleaned up.
- [ ] The offline suite passes with no keys and no network.
- [ ] Security controls are in place (read-only MCP, sanitised env, path guard).
- [ ] README, planning document and architecture diagram are complete.

---

## 14. Open questions and future work

- **Chunking quality:** move from line windows to syntax-aware function/class chunks.
- **Language coverage:** extend ingestion and `ast` analysis beyond Python.
- **Reranking:** add a lightweight reranker after vector search.
- **Evaluation harness:** build a small gold question/answer set to score retrieval.
- **Caching:** reuse embeddings for unchanged files between indexing runs.
- **Streaming:** stream final answers to reduce perceived latency.
- **Observability:** log per-stage timings (routing, retrieval, generation).

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **ADK** | Agent Development Kit, the framework that runs the agent graph. |
| **Coordinator** | The root agent that routes each request to one worker. |
| **Worker** | A specialist agent with tools and a focused instruction. |
| **MCP** | Model Context Protocol; used for read-only filesystem access. |
| **Chunk** | One overlapping line window of a source file. |
| **Provenance** | The file path and line range attached to a retrieved chunk. |
| **Grounding** | Restricting an answer to evidence returned by a tool. |
| **Run config** | ADK's per-invocation settings, including the model-call limit. |
