-- 003_add_vector_indexes.sql  : vector-search indexes for embedding columns
CREATE INDEX IF NOT EXISTS idx_symbols_code_vec ON symbols USING HNSW (embedding_code_vec) WITH (metric = 'cosine');;
CREATE INDEX IF NOT EXISTS idx_symbols_doc_vec  ON symbols USING HNSW (embedding_doc_vec) WITH (metric = 'cosine');;
CREATE INDEX IF NOT EXISTS idx_symbols_sig_vec  ON symbols USING HNSW (embedding_sig_vec) WITH (metric = 'cosine');;
