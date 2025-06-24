-- 003_add_vector_indexes.sql
SET hnsw_enable_experimental_persistence = true;

CREATE INDEX IF NOT EXISTS idx_symbols_code_vec ON symbols USING HNSW (embedding_code_vec) WITH (metric = 'cosine');;
