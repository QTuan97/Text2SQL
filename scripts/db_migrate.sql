-- Core demo tables
CREATE TABLE users (
  id        SERIAL PRIMARY KEY,
  name      TEXT NOT NULL,
  city      TEXT NOT NULL
);

CREATE TABLE orders (
  id         SERIAL PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  amount     NUMERIC(10,2) NOT NULL CHECK (amount >= 0),
  created_at TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX orders_user_id_idx     ON orders(user_id);
CREATE INDEX orders_created_at_idx  ON orders(created_at);
CREATE INDEX orders_amount_idx      ON orders(amount);

-- Optional API logs (if your app uses it)
CREATE TABLE api_logs (
  id          BIGSERIAL PRIMARY KEY,
  route       TEXT,
  method      TEXT,
  status      INTEGER,
  payload     JSONB,
  response    JSONB,
  duration_ms INTEGER,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX api_logs_created_at_idx ON api_logs(created_at);

-- Semantic/MDL support tables (overrides)
CREATE TABLE semantic_synonyms (
  phrase   TEXT PRIMARY KEY,
  maps_to  TEXT NOT NULL,
  weight   DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (weight BETWEEN 0.0 AND 1.0),
  approved BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE semantic_metrics (
  name       TEXT PRIMARY KEY,
  expression TEXT NOT NULL,          -- e.g. 'SUM(orders.amount)'
  unit       TEXT,
  synonyms   TEXT[] NOT NULL DEFAULT '{}'::text[]
);