CREATE TABLE IF NOT EXISTS semantic_synonyms (
  phrase   TEXT PRIMARY KEY,
  maps_to  TEXT NOT NULL,
  weight   DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (weight BETWEEN 0.0 AND 1.0),
  approved BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS semantic_metrics (
  name       TEXT PRIMARY KEY,
  expression TEXT NOT NULL,
  unit       TEXT,
  synonyms   TEXT[] NOT NULL DEFAULT '{}'::text[]
);