-- 004_add_doc_fts.sql  : create FTS index
PRAGMA create_fts_index('symbols', 'id', 'docstring', 'comment');
