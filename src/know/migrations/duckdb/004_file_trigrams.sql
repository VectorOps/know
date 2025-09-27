-- 004_file_trigrams.sql - filename fuzzy search support
-- Table with precomputed lowercased path and basename for cheap matching/scoring
CREATE TABLE IF NOT EXISTS files_search (
    file_id TEXT PRIMARY KEY,
    path_lc TEXT NOT NULL,
    basename_lc TEXT NOT NULL
);

-- Trigram presence index (one row per distinct trigram per file)
CREATE TABLE IF NOT EXISTS file_trigrams (
    file_id TEXT NOT NULL,
    trigram TEXT NOT NULL,
    PRIMARY KEY (file_id, trigram)
);

-- Lookup index for trigram -> file ids
CREATE INDEX IF NOT EXISTS idx_file_trigrams_trigram ON file_trigrams(trigram);
