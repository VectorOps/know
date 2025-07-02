-- Full-text index for symbols.docstring / symbols.comment
CREATE INDEX IF NOT EXISTS idx_symbols_doc_fts
    ON symbols USING fts(docstring, comment);
