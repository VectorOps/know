-- 003_node_boost.sql - add precomputed search boost column to nodes
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS search_boost DOUBLE;
ALTER TABLE nodes ALTER COLUMN search_boost SET DEFAULT 1.0;
UPDATE nodes SET search_boost = 1.0 WHERE search_boost IS NULL;
