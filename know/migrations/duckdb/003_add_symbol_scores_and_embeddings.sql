-- Extend “symbols” table with quality-score and embedding columns.

-- Quality scores
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS score_lint        DOUBLE;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS score_complexity  INTEGER;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS score_coverage    DOUBLE;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS score_security_flags TEXT;   -- JSON-encoded list[str]

-- Embeddings
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS embedding_code_vec DOUBLE[];
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS embedding_doc_vec  DOUBLE[];
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS embedding_sig_vec  DOUBLE[];
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS embedding_model    TEXT;
