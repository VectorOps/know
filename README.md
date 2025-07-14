
# VectorOps Know

**VectorOps Know** is an advanced toolkit for semantic codebase analysis and developer productivity. It provides utilities for symbol search, file summarization, repository relationship mapping, and more—leveraging code embeddings and structural parsing for actionable insights in projects of significant scale and complexity.

## Features

- Semantic code and symbol search
- File and project summarization using embeddings
- Relationship and impact analysis via code graph traversal
- Pluggable architecture for multiple languages and extensions
- User-friendly command-line tools and Python API

## Installation

Clone the repository and install dependencies:

```shell
git clone <repository_url>
cd <repository_dir>
pip install -r requirements.txt
```

## Example CLI Usage

### 1. Symbol Search CLI

Interactively search for symbols with text or natural-language queries:

```sh
python tools/searchcli.py --path .
```

- You will enter an interactive prompt.  
  Type your search (e.g., `authentication` or `def my_function`) and press Enter.
- Exit the prompt with `/exit` or `Ctrl-D`.

### 2. File Summarization CLI

Print a summary for one or more files:

```sh
python tools/filesummarycli.py --path . path/to/file1.py path/to/file2.py
```

- Replace `path/to/file1.py`, etc. with one or more files relative to your project root.
- Optional: Control summary detail level with `-m` (e.g., `-m FullSummary`). Example:

```sh
python tools/filesummarycli.py --path . -m FullSummary my_module.py
```

### 3. RepoMap CLI (Repository Relationship Explorer)

Launch an interactive repo map tool for exploring file and symbol relationships:

```sh
python tools/repomapcli.py --path .
```

- This opens an interactive prompt.
- Use `/help` for available commands (such as `/sym <name>`, `/file <rel/path.py>`, `/prompt <text>`, `/run`).

**Example session:**
```
> /prompt validate my_function user input
> /run
```

### 4. Conversational Chat CLI

Launch an LLM-powered chat to ask questions about your codebase:

```sh
OPENAI_API_KEY=your-key python tools/chatcli.py --path .
```

- This opens an interactive chat.
- Type your requests or `/exit` to leave.
- Use `--model` to select the LLM model (default: `gpt-4.1`).

**Example:**
```
> How is database authentication implemented?
```

**Tips:**
- For all commands, replace `--path .` with your project’s root directory if not running from the top level.
- Add `--enable-embeddings` to any command to unlock semantic searching and summarization (if you have embedding models installed).

## Python API Example

You can also use VectorOps Know programmatically:

```python
from know.project import init_project, ProjectSettings
from know.tools.symbolsearch import SearchSymbolsTool

settings = ProjectSettings(project_path=".")
project = init_project(settings)
results = SearchSymbolsTool().execute(project, query="main")
for r in results:
    print(r.name, r.fqn, r.kind)
```

## Getting Started

- Use the `--help` flag on any CLI tool for further details on arguments and configuration:

  ```sh
  python tools/searchcli.py --help
  ```

- The toolkit is extensible: you can add new languages, parsing strategies, or analysis tools by following the established interfaces.

## License

Apache 2.0 License
