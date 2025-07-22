-- 001_init.sql  : create initial schema
CREATE TABLE IF NOT EXISTS repos (
    id TEXT PRIMARY KEY,
    name TEXT,
    root_path TEXT,
    remote_url TEXT,
    default_branch TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS packages (
    id TEXT PRIMARY KEY,
    name TEXT,
    repo_id TEXT,
    language TEXT,
    virtual_path TEXT,
    physical_path TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    package_id TEXT,
    path TEXT,
    file_hash TEXT,
    last_updated DOUBLE,
    commit_hash TEXT,
    mime_type TEXT,
    language TEXT,
    metrics_total_loc INTEGER,
    metrics_code_loc INTEGER,
    metrics_comment_loc INTEGER,
    metrics_cyclomatic_complexity INTEGER
);

CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    file_id TEXT,
    package_id TEXT,
    name TEXT,
    fqn TEXT,
    body BLOB,
    kind TEXT,
    parent_symbol_id TEXT,
    start_line INTEGER,
    start_col INTEGER,
    end_line INTEGER,
    end_col INTEGER,
    start_byte INTEGER,
    end_byte INTEGER,
    visibility TEXT,
    modifiers JSON,
    docstring TEXT,
    comment TEXT,
    exported BOOLEAN,
    signature JSON,
    score_lint DOUBLE,
    score_complexity INTEGER,
    score_coverage DOUBLE,
    score_security_flags TEXT[],
    embedding_code_vec FLOAT[1024],
    embedding_model TEXT
);

CREATE TABLE IF NOT EXISTS import_edges (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    from_package_id TEXT,
    from_file_id TEXT,
    to_package_physical_path TEXT,
    to_package_virtual_path TEXT,
    to_package_id TEXT,
    alias TEXT,
    dot BOOLEAN,
    external BOOLEAN,
    raw TEXT
);
