# Know

This project provides in-memory and DuckDB-backed repositories for managing metadata about code repositories, packages, files, symbols, and import edges.

## Features

- In-memory data repository for fast prototyping and testing.
- DuckDB-backed data repository with automatic SQL schema migrations.
- Pydantic models for structured metadata.
- Repository interfaces for CRUD operations and specialized queries.

## Installation

Add the package dependencies including DuckDB:

```bash
pip install duckdb pydantic pytest
```

## Usage

Import and instantiate the desired data repository:

```python
from know.stores.memory import InMemoryDataRepository
from know.stores.duckdb import DuckDBDataRepository

# In-memory repository
data_repo = InMemoryDataRepository()

# DuckDB repository (file-based)
data_repo = DuckDBDataRepository(db_path="path/to/db.duckdb")
```

## Migrations

DuckDB repository automatically applies SQL migrations from the `know.migrations` package on initialization.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## License

TODO: Add license information.
