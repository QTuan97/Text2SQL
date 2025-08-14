CREATE TABLE IF NOT EXISTS nl2sql_lessons (
  id           BIGSERIAL PRIMARY KEY,
  question     TEXT NOT NULL,
  sql          TEXT NOT NULL,
  tables_used  TEXT[] NOT NULL DEFAULT '{}'::text[],
  vector       JSONB NOT NULL,
  successes    INTEGER NOT NULL DEFAULT 1,
  last_seen    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  schema_sig   TEXT
);

CREATE INDEX IF NOT EXISTS nl2sql_lessons_qts ON nl2sql_lessons USING GIN (to_tsvector('simple', question));
CREATE INDEX IF NOT EXISTS nl2sql_lessons_seen ON nl2sql_lessons (last_seen DESC);