-- Adds per-file metrics columns that already exist in `FileMetadata`

ALTER TABLE files ADD COLUMN IF NOT EXISTS metrics_total_loc              INTEGER;
ALTER TABLE files ADD COLUMN IF NOT EXISTS metrics_code_loc               INTEGER;
ALTER TABLE files ADD COLUMN IF NOT EXISTS metrics_comment_loc            INTEGER;
ALTER TABLE files ADD COLUMN IF NOT EXISTS metrics_cyclomatic_complexity  INTEGER;
