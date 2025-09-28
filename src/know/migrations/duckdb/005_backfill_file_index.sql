-- 005_backfill_file_index.sql - backfill files_search and file_trigrams from existing files

-- Backfill files_search rows for files missing an index
INSERT INTO files_search (file_id, path_lc, basename_lc)
SELECT
    f.id AS file_id,
    lower(f.path) AS path_lc,
    lower(regexp_extract(f.path, '([^/]+)$', 1)) AS basename_lc
FROM files f
LEFT JOIN files_search fs ON fs.file_id = f.id
WHERE fs.file_id IS NULL;

-- Backfill file_trigrams rows for indexed files missing trigrams
INSERT INTO file_trigrams (file_id, trigram)
SELECT t.file_id, t.tri
FROM (
    SELECT DISTINCT
        fs.file_id,
        substring(fs.path_lc, gs.i, 3) AS tri
    FROM files_search fs
    CROSS JOIN generate_series(1, length(fs.path_lc) - 2) AS gs(i)
    WHERE length(fs.path_lc) >= 3
) AS t
LEFT JOIN file_trigrams ft
  ON ft.file_id = t.file_id AND ft.trigram = t.tri
WHERE ft.file_id IS NULL;
