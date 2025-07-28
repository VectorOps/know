# VectorOps – *Know*

VectorOps *Know* is an extensible code-intelligence helper library. It scans your repository, builds a language-aware graph of files / packages / symbols and exposes high-level tooling for search, summarisation, ranking and graph analysis.
All functionality is available from the command line, through a clean Python API, or via a lightweight **MCP** (Machine-Comprehension Provider) micro-service for chat/LLM workflows.

## Key Features
• Multi-language parsing (Python, TypeScript, Go, …)  
• Local (in-memory) or DuckDB metadata store  
• Optional sentence-transformer embeddings for semantic search  
• Rich tool catalogue automatically exported as OpenAI JSON schemas  
• Ready-to-run FastMCP server with zero configuration  

---

## Installation
```bash
git clone https://github.com/<you>/vectorops.git
cd vectorops
pip install -r requirements.txt
```

> **Embeddings**  
> To enable semantic search, install a sentence-transformer model (e.g. `pip install sentence-transformers`) and start tools with `--enable-embeddings`.

---

## Built-in Tools

| Tool name (API)          | Python class                                    | CLI helper                 | Purpose |
|--------------------------|-------------------------------------------------|----------------------------|---------|
| `vectorops_list_files`   | `know.tools.filelist.ListFilesTool`             | – *(used via API/MCP)*     | Return files whose paths match glob patterns |
| `vectorops_summarize_files` | `know.tools.filesummary.SummarizeFilesTool` | `tools/filesummarycli.py`  | Create import & symbol summaries for files |
| `vectorops_search`       | `know.tools.nodesearch.NodeSearchTool`          | `tools/searchcli.py`       | Hybrid (text + vector) symbol search |
| `vectorops_repomap`      | `know.tools.repomap.RepoMapTool`                | `tools/repomapcli.py`      | Rank files with Random-Walk-with-Restart on the code graph |

All tools inherit `BaseTool`.  When the MCP server is started, each tool becomes an HTTP endpoint and is also advertised through an OpenAI‐compatible schema at `/openai.json`.

---

## Quick CLI Examples

```bash
# 1 – Search for a class or function
python tools/searchcli.py --path . "vector search"

# 2 – Summarise a file
python tools/filesummarycli.py --path . know/project.py -m summary_full

# 3 – Generate a repo relevance map
python tools/repomapcli.py --path . --prompt "authentication token"
```

---

## MCP Server

Spin up the FastMCP server and expose every tool via HTTP:

```bash
python tools/mcpserver.py \
    --project-path . \
    --mcp-host 0.0.0.0 --mcp-port 8080
```

Example request:
```bash
curl -X POST http://localhost:8080/vectorops/vectorops_search \
     -H "Content-Type: application/json" \
     -d '{"symbol_name":"init_project","limit":5}'
```
If you started the server with `--mcp-auth-token`, include `Authorization: Bearer <token>`.

---

## Using the Python API

```python
from know.project import init_project, ProjectSettings
from know.tools.nodesearch import NodeSearchTool
from know.tools.repomap   import RepoMapTool
from know.data import NodeSearchQuery

# 1. bootstrap and scan project
settings = ProjectSettings(project_path=".")
project  = init_project(settings)      # first run performs a full scan

# 2. run a symbol search
req  = NodeSearchTool.tool_input(symbol_name="refresh", limit=10)
hits = NodeSearchTool().execute(project, req)
for h in hits:
    print(h.name, h.kind, h.file_path)

# 3. create a repo map seeded via prompt
map_req = RepoMapTool.tool_input(prompt="database connection pooling", limit=15)
for item in RepoMapTool().execute(project, map_req):
    print(f"{item.score:0.4f}", item.file_path)

# 4. low-level access to repositories
repo_id = project.get_repo().id
symbols = project.data_repository.symbol.search(
    repo_id=repo_id,
    query=NodeSearchQuery(symbol_name="Project")
)
print("Found", len(symbols), "symbols named Project")

project.destroy()                      # graceful shutdown
```

---

## Extending Know

1. **Parsers** – implement `AbstractCodeParser` and register via `CodeParserRegistry`.  
2. **Tools**   – subclass `BaseTool`; registration is automatic.  
3. **Components** – derive from `ProjectComponent` and register with `Project.register_component`.

---

## License

VectorOps Know is released under the **Apache 2.0** license.  
See `LICENSE.txt` for the full text.
