-- Full-text index for symbols.docstring / symbols.comment
PRAGMA create_fts_index('symbols',
                        'idx_symbols_doc_fts',
                        'docstring',
                        'comment');
