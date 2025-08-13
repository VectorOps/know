-- 003_node_boost.sql - add precomputed search boost column to nodes
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS search_boost DOUBLE;
ALTER TABLE nodes ALTER COLUMN search_boost SET DEFAULT 1.0;
UPDATE nodes
SET search_boost = CASE kind
    WHEN 'function' THEN 2.0
    WHEN 'method' THEN 2.0
    WHEN 'method_def' THEN 2.0
    WHEN 'class' THEN 1.5
    WHEN 'property' THEN 1.3
    WHEN 'literal' THEN 0.9
    ELSE 1.0
END
WHERE search_boost IS NULL;
