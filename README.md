# VectorOps - *Know*

VectorOps *Know* is an extensible code-intelligence helper library. It scans your repository, builds a language-aware graph of files / packages / symbols and exposes high-level tooling for search, summarisation, ranking and graph analysis.
All functionality is available from the command line, through a clean Python API, or via a lightweight **MCP** (Model Communication Protocol) service for chat/LLM workflows.

## Why?
It is not feasible to pluck all files of a large project to an LLM context and call it a day. Know will try to help LLM to find relevant files, methods, functions, documentation snippets to be able to solve a task at hand.

## Key Features

 * Multi-language parsing (Python, Go, JavaScript, TypeScript, Java, ...)
 * Local in-memory or on-disk DuckDB metadata store
 * Optional sentence-transformer embeddings for semantic search
 * Rich tool catalogue automatically exported as OpenAI JSON schemas
 * Ready-to-run FastMCP server with zero configuration

---

## Installation

The project is in active development and an official PyPI release is planned very soon.

Disclaimer: This was not tested on Windows, but Windows support is coming.

For now, clone the repo and use `uv` to play with the tools:

```bash
git clone https://github.com/vectorops/know.git
cd know
uv sync
```

## Settings

*Know* exposes all settings via Pydantic BaseModel and it is possible to use  [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to manage configuration. This provides a flexible way to configure the project through environment variables, JSON files, or command-line arguments.

### Key Settings

There are three main settings you'll almost always need to provide:

*   `--project-name`: A name for your project, which represents a collection of multiple source code repositories. This field is required.
*   `--repo-name`: The name of the repository you are currently working with. This field is required.
*   `--repo-path`: The local filesystem path to the root of the repository. This is required when first adding a repository to a project.

Conceptually, multiple repositories can be part of a project. Search tool will look for things in all repositories, but boost local results. Repomap only keep track of current repository.

By default parsed results won't persist to disk. Pass the `--repository-connection [filename]` to persist database between tool restarts.

### Examples

**Using CLI flags:**

All CLI tools accept settings as command-line arguments.

```bash
# Scan a repository and run a search
uv run python tools/searchcli.py \
    --project-name="my-org/know" \
    --repo-name="know" \
    --repo-path="."
```

**Using Environment Variables:**

This is particularly useful when running the MCP server.

```bash
# Set environment variables
export KNOW_PROJECT_NAME="my-org/know"
export KNOW_REPO_NAME="know"
export KNOW_REPO_PATH="."
export KNOW_EMBEDDING_ENABLED=true
export KNOW_EMBEDDING_MODEL_NAME="BAAI/bge-code-v1"

# Run a tool (no need to pass settings as flags)
uv run python tools/searchcli.py
```

## Embeddings
To enable semantic search, pick a sentence-transformer model and start tools with `--embedding.enable true`. Provide model to use via `--embedding.model-name`, such as `--embedding.model-name BAAI/bge-code-v1`.

It is highly recommended that embeddings are cached and persisted across runs to save on computing costs. To enable caching, pass `--embedding.cache-backend duckdb --embedding.cache-path cache.duckdb`.

---

## Built-in Tools

| Tool name (API)          | Python class                                    | CLI helper                 | Purpose |
|--------------------------|-------------------------------------------------|----------------------------|---------|
| `vectorops_list_files`   | `know.tools.filelist.ListFilesTool`             | - *(used via API/MCP)*     | Return files whose paths match glob patterns |
| `vectorops_summarize_files` | `know.tools.filesummary.SummarizeFilesTool` | `tools/filesummarycli.py`  | Create import & symbol summaries for files |
| `vectorops_search`       | `know.tools.nodesearch.NodeSearchTool`          | `tools/searchcli.py`       | Hybrid (text + vector) symbol search |
| `vectorops_repomap`      | `know.tools.repomap.RepoMapTool`                | `tools/repomapcli.py`      | Rank files with Random-Walk-with-Restart on the code graph |

All tools inherit `BaseTool`.  When the MCP server is started, each tool becomes an HTTP endpoint and is also advertised through an OpenAI-compatible schema at `/openai.json`.

---

## Quick CLI Examples

```bash
# 1 - Search for a class or function
uv run python tools/searchcli.py --project-name "prj" --repo-name "know" --repo-path .

# 2 - Summarise a file
uv run python tools/filesummarycli.py --project-name "prj" --repo-name "know" --repo-path . --file-paths know/project.py --summary-mode summary_full

# 3 - Generate a repo relevance map
uv run python tools/repomapcli.py --project-name "prj" --repo-name "know" --repo-path .
```

---

## MCP Server

To run the MCP server, you first need to install the optional `mcp` dependencies:
```bash
uv sync --extra mcp
```

The server is configured via an `mcp.json` file in the project root, or via environment variables (with a `KNOW_` prefix). Here is an example `mcp.json`:

```json
{
    "project_name": "test",
    "repo_name": "know",
    "repo_path": "."
}
```

**Development Server**

Spin up a development server with hot-reloading:
```bash
uv run fastmcp dev tools/mcpserver.py:mcp
```
The app will be available at `http://localhost:8000`.

**Production Server**

To run the streaming-capable production server:
```bash
uv run python tools/mcpserver.py
```
This will start the server on `http://localhost:8000` by default. You can change host and port via settings (e.g., `KNOW_MCP_HOST`, `KNOW_MCP_PORT` or in `mcp.json`).

Example request:
```bash
curl -X POST http://localhost:8000/vectorops/vectorops_search \
     -H "Content-Type: application/json" \
     -d '{"query":"init_project","limit":5}'
```

By default the server is unprotected, refer to [FastMCP documentation](https://gofastmcp.com/servers/auth/authentication#environment-configuration) to find out how to enable authentication.

---

## Using the Python API

```python
from know import init_project
from know.settings ProjectSettings
from know.tools.nodesearch import NodeSearchTool
from know.tools.repomap import RepoMapTool
from know.data import NodeSearchQuery

# 1. bootstrap and scan project
settings = ProjectSettings(project_name="prj", repo_name="know", repo_path=".")
project  = init_project(settings)

# 2. run a symbol search
req  = NodeSearchTool.tool_input(symbol_name="refresh", limit=10)
hits = NodeSearchTool().execute(project, req)
for h in hits:
    print(h.name, h.kind, h.file_path)

# 3. create a repo map seeded via prompt
map_req = RepoMapTool.tool_input(prompt="duckdb.py search", limit=15)
for item in RepoMapTool().execute(project, map_req):
    print(f"{item.score:0.4f}", item.file_path)

# 4. low-level access to repositories
repo_id = project.default_repo.id
symbols = project.data_repository.node.search(NodeSearchQuery(repo_ids=[repo_id], symbol_name="Project"))
print("Found", len(symbols), "symbols named Project")

project.destroy()
```

---

## Extending Know

1. **Parsers** - implement `AbstractCodeParser` and register via `CodeParserRegistry`.  
2. **Tools**   - subclass `BaseTool`; registration is automatic.  
3. **Components** - derive from `ProjectComponent` and register with `Project.register_component`.

---

## License

VectorOps Know is released under the **Apache 2.0** license.  
See `LICENSE.txt` for the full text.
