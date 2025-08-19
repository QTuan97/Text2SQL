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