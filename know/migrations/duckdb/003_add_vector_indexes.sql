-- 003_add_vector_indexes.sql  : vector-search indexes for embedding columns
INSTALL vss;
LOAD vss;

CREATE INDEX IF NOT EXISTS idx_symbols_code_vec ON symbols USING vss(embedding_code_vec);
CREATE INDEX IF NOT EXISTS idx_symbols_doc_vec  ON symbols USING vss(embedding_doc_vec);
CREATE INDEX IF NOT EXISTS idx_symbols_sig_vec  ON symbols USING vss(embedding_sig_vec);
