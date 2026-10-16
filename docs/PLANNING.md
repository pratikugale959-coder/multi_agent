# Planning Document — Multi-Agent Code Helper

> **What this is.** The plan behind this project: the problem, the requirements, the
> decisions and their rationale, the phased build order, the risks, and the
> acceptance criteria — followed by the outcome for each item.
>
> **Companion documents.** Technical reference: [`README.md`](../README.md) ·
> Architecture diagram: [`architecture.drawio`](architecture.drawio) (open at
> [app.diagrams.net](https://app.diagrams.net)).

- **Project:** Multi-Agent Code Helper
- **Prepared:** 2026-10-16
- **Status:** ✅ Complete — all planned phases delivered and verified

---

## 1. Context and problem statement

A developer joining an unfamiliar Python repository spends a long time answering four
recurring questions:

1. *Where* is a given feature implemented?
2. *What* does a particular file do?
3. *How* do the modules connect (imports, dependencies, entry points)?
4. *How* do I install, run and navigate the project?

Generic chatbots answer these questions from memory and invent filenames and APIs. The
goal is an assistant that answers **only from the actual repository**, cites the file and
line range, and can be verified.

**Problem statement:** build a multi-agent assistant that answers the four questions above
from a real, unfamiliar Python codebase, with every claim traceable to evidence, using
Google ADK, NVIDIA NIM and Pinecone — implemented without a RAG framework.

---

## 2. Objectives and success criteria

| # | Objective | Success criterion | Outcome |
|---|---|---|---|
| O1 | Map a question to the right specialist | Coordinator routes to the correct worker for each of the 4 question types | ✅ verified (`scripts/run_agent_demo.py`) |
| O2 | Ground every answer in the repo | Answers cite a real file + line range, or state "no evidence" | ✅ sample Q1–Q3 in README §10 |
| O3 | Implement RAG by hand | Ingestion→chunking→embeddings→store→retrieval as separate, testable modules | ✅ README §4 |
| O4 | Deterministic analysis where possible | ≥1 tool per worker; several use no AI at all | ✅ 4 of 5 tools are AI-free |
| O5 | One shared chat model | All five agents use a single cached model instance | ✅ `config.get_chat_model()` |
| O6 | Offline-testable | Full test suite runs with no keys and no network | ✅ 118 passed |
| O7 | Safe by default | Read-only file access, sanitised subprocess env, no secret leakage | ✅ README §11 |

---

## 3. Scope

**In scope**
- Google ADK coordinator + four specialist worker agents.
- A hand-written RAG pipeline (no LangChain / LlamaIndex).
- NVIDIA NIM embeddings and chat via LiteLLM.
- Pinecone serverless vector storage.
- A read-only Filesystem MCP integration.
- Deterministic `ast`-based architecture analysis.
- An offline pytest suite and a small set of live integration scripts.
- Documentation (README + draw.io architecture map + this plan).

**Out of scope (deliberately)**
- A custom website — the ADK terminal (`adk run`) and web (`adk web`) interfaces are used.
- Supporting languages other than Python.
- LLM-based chunking or graph databases.
- Destructive vector-store operations (the code never deletes the index).
- Multi-hop agent conversations (routing is a single `coordinator → worker` hop).

---

## 4. Requirements

**Functional**
- F1 Route each of the four question types to the right worker.
- F2 Semantic search over indexed source code with provenance (file + line range).
- F3 Read and explain a complete file, preferably via MCP.
- F4 Parse real imports/definitions with `ast` (not a model guess).
- F5 Provide a project overview: structure, entry points, documented run steps.
- F6 Index a repository from the command line (`python -m code_helper.index_codebase`).

**Non-functional**
- N1 Deterministic core (same input → same analysis; no AI in the `ast` layer).
- N2 Offline testability (no keys/network needed for the suite).
- N3 Security (read-only FS access; secrets never surfaced or inherited by child processes).
- N4 Clear failures (missing credentials raise before any network call).
- N5 Reproducibility (pinned model ids; typed, validated configuration).
- N6 Readability (one module per RAG stage; documented public functions).

---

## 5. Approach and key design decisions

| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| D1 | **ADK coordinator + 4 workers**, workers cannot transfer to each other | Keeps routing a single hop and answers attributable to one worker | Free-form multi-agent hand-off caused confused meta-answers (see R4) |
| D2 | **Coordinator has no tools** | The Boss cannot answer directly, so every answer is grounded in a worker's tool output | Giving the coordinator tools blurred responsibility |
| D3 | **Custom RAG, stage-per-module** | Demonstrates the pipeline explicitly and is unit-testable in isolation | LangChain/LlamaIndex hid the stages and added heavy dependencies |
| D4 | **Deterministic `ast` tools** for architecture | Imports/dependencies are facts; parsing beats guessing | Asking the LLM to infer dependencies was unreliable |
| D5 | **One shared chat model** (`get_chat_model`) | Explicit assignment requirement; one place to configure | Per-agent models risked inconsistent behaviour |
| D6 | **Separate embedding config** from chat config | Prevents accidentally swapping the chat model for the embedding model | Sharing one "model" setting was error-prone |
| D7 | **Typed `Settings`** with fail-fast credential guards | One validated source of truth; clear errors | Scattered `os.getenv` calls across modules |
| D8 | **Line-based chunking** (100/20) | Simple, deterministic, no extra dependencies | Semantic/AST chunking added complexity beyond scope |
| D9 | **Local JSON fallback** considered for the pipeline | Keeps demos runnable without a Pinecone key | — |
| D10 | **Filesystem MCP, restricted** (root = target project, read-only allow-list, sanitised env) | Real MCP integration without giving the model write access or our secrets | Full MCP toolset was unsafe |

---

## 6. Planned architecture

The system is planned as five layers, mapped in
[`architecture.drawio`](architecture.drawio):

```
Clients & Entry Points
        │
Agent Layer (Google ADK)   coordinator → 4 workers
        │
Tools                      rag_search · read_project_file / MCP · list+inspect · overview
        │
RAG Pipeline               ingestion → chunking → embeddings → vector_store → retrieval → rag_pipeline
        │
External Services          NVIDIA NIM (chat + embeddings) · Pinecone · Filesystem MCP server
```

Cross-cutting: `config.py` (typed settings), `paths.py` (path guard),
`mcp_filesystem.py` (read-only allow-list), and the shared chat model.

---

## 7. Phased implementation plan

Build order was bottom-up: infrastructure and the pipeline first, agents and tools on top,
then verification and documentation.

| Phase | Deliverable | Modules | Status |
|---|---|---|---|
| P0 | Reconnaissance: confirm models, providers, dims | — | ✅ |
| P1 | Configuration and secrets handling | `config.py`, `paths.py` | ✅ |
| P2 | Stage 1–2: ingestion and chunking | `ingestion.py`, `chunking.py` | ✅ |
| P3 | Stage 3: NVIDIA NIM embeddings | `embeddings.py` | ✅ |
| P4 | Stage 4: Pinecone vector store | `vector_store.py` | ✅ |
| P5 | Stage 5: retrieval and formatting | `retrieval.py` | ✅ |
| P6 | Pipeline orchestration + index CLI | `rag_pipeline.py`, `index_codebase.py` | ✅ |
| P7 | Agents: coordinator + four workers | `agent.py`, `agents.py` | ✅ |
| P8 | The five custom ADK tools | `tools/*.py` | ✅ |
| P9 | Filesystem MCP integration | `mcp_filesystem.py` | ✅ |
| P10 | Offline test suite | `tests/` (118 tests) | ✅ |
| P11 | Live integration verification | `scripts/*.py` | ✅ |
| P12 | Documentation + architecture diagram | `README.md`, `docs/` | ✅ |

**Evidence of completion** is in README §8 (verified runs): 118 offline tests passed;
NVIDIA NIM embedding dimension verified = 2048; Pinecone index `code-helper` created and
queried; Filesystem MCP 10-tool allow-list confirmed against the live server; ADK terminal
and web interfaces launched.

---

## 8. Agent and tool plan

| Agent | Type | Responsibility | Tool(s) | AI used? |
|---|---|---|---|---|
| `coordinator` | root | Route to one worker; return its answer | *(none)* | routing only |
| `code_searcher` | worker | Where is X implemented? | `rag_search` | yes |
| `code_explainer` | worker | What does a whole file do? | Filesystem MCP + `read_project_file` | optional |
| `architecture_mapper` | worker | Imports, dependencies, entry points | `list_project_files`, `inspect_imports` | no |
| `onboarding_guide` | worker | Purpose, setup, reading order | `get_project_overview` | no |

**Plan rule (met):** every worker has at least one tool, and at least one tool operates with
no AI. In practice four of the five tools are fully deterministic.

---

## 9. RAG pipeline plan

| Stage | Module | Plan |
|---|---|---|
| 1 · Ingestion | `ingestion.py` | Walk the target; skip `.git`, `.venv`, caches, secrets, binary and oversized files; keep text + relative path |
| 2 · Chunking | `chunking.py` | Custom line window: size 100, overlap 20; stable ids `file:start-end`; carry file, line range, text |
| 3 · Embeddings | `embeddings.py` | `nvidia/nemotron-3-embed-1b` via LiteLLM; batching, retries, **dimension validation** |
| 4 · Storage | `vector_store.py` | Pinecone serverless, cosine, 2048-dim; deterministic vector ids; metadata for provenance |
| 5 · Retrieval | `retrieval.py` | Embed the query with the **same** model; return top-k with scores and formatting |
| 6 · Orchestration | `rag_pipeline.py` | `index_repository()` (1–4) and `answer_question()`; `IndexReport` summary |

**Chunking trade-off accepted:** a chunk may begin mid-function, so retrieved text is a
*pointer*, not a self-contained unit. Overlap mitigates but does not remove this (documented
in README §4 "Limitations").

---

## 10. Security plan

| Concern | Planned control |
|---|---|
| Secrets at rest | Keys only in `.env` (git-ignored); `.env.example` holds placeholders |
| Secrets in diagnostics | `describe_configuration()` / `describe_mcp_configuration()` return redacted views |
| Missing keys | Fail fast with `ConfigError` **before** any network call |
| Path traversal | `paths.py` confines all file access to the target repo |
| File access | MCP exposes only read tools (`read_text_file`, `list_directory`, …) — never write |
| Child process | `npx` receives a sanitised, allow-listed environment (no `*_KEY`, `*_TOKEN`, …) |
| Vector data | Code never deletes the index or vectors |

---

## 11. Testing strategy

- **Offline first.** All external services are replaced by fakes, so the suite needs no keys
  and no network (README §8, `tests/conftest.py`).
- **One test module per stage**, plus agents, tools, config and MCP:
  ingestion, chunking, embeddings, vector store, retrieval, pipeline, agents, tools, config,
  MCP (11 modules).
- **Determinism guarded by tests**: chunk boundaries/overlap, frozen dataclasses, tool
  signatures, read-only tool filtering, secret stripping in the MCP env.
- **Live integration is separate**: `scripts/*.py` make real calls and are *not* part of
  pytest. Result: **118 passed**.

---

## 12. Risks and mitigations

| # | Risk | Likelihood | Mitigation | Result |
|---|---|---|---|---|
| R1 | Model returns an unexpected argument type (e.g. `top_k="5"`) | Medium | Coerce and validate tool inputs | Fixed — README §9 Problem 1 |
| R2 | Internal `settings` param leaks into a tool and the model fills it in | Medium | Public tools take only user-facing args; resolve config via a `get_settings` seam | Fixed — README §9 Problem 2 |
| R3 | Embedding dimension mismatch vs. Pinecone index | Medium | Validate returned dimension; verify live (2048) | Verified |
| R4 | Workers delegate to each other → confused answers | Medium | `disallow_transfer_to_parent` / `disallow_transfer_to_peers` | Fixed — README §9 Problem 3 |
| R5 | MCP gives write access or inherits secrets | Low | Read-only filter + sanitised env + confined root | Mitigated |
| R6 | Missing keys cause obscure SDK errors | Medium | Fail-fast `ConfigError` guards | Mitigated |
| R7 | Line-based chunking yields low-information chunks | Medium | Documented as a known limitation | Accepted (README §4, §12) |
| R8 | Model adds a claim a tool did not report | Medium | Instructions forbid it; tool output treated as ground truth | Documented (README §12) |
| R9 | Secrets accidentally published | Low | `.env` git-ignored; `.env.example` scrubbed to placeholders before first commit | Mitigated |

---

## 13. Deliverables

1. `code_helper/` — the implementation (config, 5-stage RAG modules, orchestration, agents, 5 tools, MCP, path guard, index CLI).
2. `tests/` — the offline suite (118 tests).
3. `scripts/` — live integration checks.
4. `target_project/` — the repository analysed by the agents (kept separate).
5. `README.md` — full technical reference (verified results, problems/fixes, security).
6. `docs/architecture.drawio` — the architecture map.
7. `docs/PLANNING.md` — this document.
8. `requirements.txt`, `pytest.ini`, `ruff.toml`, `.gitignore`, `.env.example`.

---

## 14. Acceptance criteria (definition of done)

- [x] Coordinator routes all four question types to the correct worker.
- [x] Answers cite a real file and line range, or explicitly report no evidence.
- [x] RAG implemented by hand as five separate, documented modules.
- [x] Five tools defined; ≥1 deterministic (four are AI-free).
- [x] All five agents share one cached chat model; embedding config kept separate.
- [x] Offline test suite passes with no keys/network (**118 passed**).
- [x] Live checks verify embeddings (2048), Pinecone, MCP, and agent routing.
- [x] Security controls in place (read-only FS, sanitised env, redacted diagnostics).
- [x] Documentation complete, including the draw.io architecture diagram.

---

## 15. Open questions and future work

- **Chunking quality:** move from line-based to syntax-aware (e.g. function/class) chunking.
- **Language coverage:** extend `ast` analysis and ingestion to more languages.
- **Answer faithfulness:** add a citation-checking step to reject claims without evidence.
- **Evaluation harness:** add a small gold Q&A set to score retrieval/answer quality.
- **Cost/latency:** cache embeddings for unchanged files between index runs.
- **Optional UI:** a thin web front-end over the ADK API (currently out of scope by design).

---

## 16. Revision history

| Date | Revision | Note |
|---|---|---|
| 2026-10-16 | 1.0 | Initial planning document, written after delivery to capture the plan and its outcome |
