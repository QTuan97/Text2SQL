-- Stores “virtual datasets” defined by users via UI
CREATE TABLE IF NOT EXISTS custom_datasets (
  id           SERIAL PRIMARY KEY,
  name         TEXT UNIQUE NOT NULL,
  description  TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Each dataset has many logical tables
CREATE TABLE IF NOT EXISTS custom_tables (
  id          SERIAL PRIMARY KEY,
  dataset_id  INTEGER NOT NULL REFERENCES custom_datasets(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  UNIQUE(dataset_id, name)
);

-- Column definitions for those tables
CREATE TABLE IF NOT EXISTS custom_columns (
  id          SERIAL PRIMARY KEY,
  table_id    INTEGER NOT NULL REFERENCES custom_tables(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  pg_type     TEXT NOT NULL,
  nullable    BOOLEAN NOT NULL DEFAULT TRUE,
  is_pk       BOOLEAN NOT NULL DEFAULT FALSE,
  fk_table    TEXT,
  fk_column   TEXT,
  description TEXT,
  synonyms    TEXT[] NOT NULL DEFAULT '{}'::text[],
  UNIQUE(table_id, name)
);