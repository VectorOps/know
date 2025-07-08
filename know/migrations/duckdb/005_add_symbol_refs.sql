-- 005_add_symbol_refs.sql  : create table for SymbolRef
CREATE TABLE IF NOT EXISTS symbol_refs (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    name TEXT NOT NULL,
    raw TEXT NOT NULL,
    type TEXT NOT NULL,
    to_package_id TEXT
);
