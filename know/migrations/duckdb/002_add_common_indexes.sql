-- Indexes frequently used by repository read-operations
-- (all guarded with IF NOT EXISTS for repeated runs)

-- repos
CREATE UNIQUE INDEX IF NOT EXISTS idx_repos_id          ON repos(id);
CREATE        INDEX IF NOT EXISTS idx_repos_root_path   ON repos(root_path);

-- packages
CREATE UNIQUE INDEX IF NOT EXISTS idx_packages_id       ON packages(id);
CREATE        INDEX IF NOT EXISTS idx_packages_physpath ON packages(physical_path);

-- files
CREATE UNIQUE INDEX IF NOT EXISTS idx_files_id          ON files(id);
CREATE        INDEX IF NOT EXISTS idx_files_path        ON files(path);
CREATE        INDEX IF NOT EXISTS idx_files_repo_id     ON files(repo_id);
CREATE        INDEX IF NOT EXISTS idx_files_package_id  ON files(package_id);

-- symbols
CREATE UNIQUE INDEX IF NOT EXISTS idx_symbols_id        ON symbols(id);
CREATE        INDEX IF NOT EXISTS idx_symbols_file_id   ON symbols(file_id);

-- import_edges
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_id          ON import_edges(id);
CREATE        INDEX IF NOT EXISTS idx_edges_from_pkg_id ON import_edges(from_package_id);
