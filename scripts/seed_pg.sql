CREATE TABLE IF NOT EXISTS users (
  id   INT PRIMARY KEY,
  name TEXT NOT NULL,
  city TEXT
);

CREATE TABLE IF NOT EXISTS orders (
  id        INT PRIMARY KEY,
  user_id   INT NOT NULL REFERENCES users(id),
  amount    NUMERIC(10,2),
  created_at TIMESTAMP DEFAULT NOW()
);

-- seed (safe to rerun)
INSERT INTO users (id,name,city) VALUES
  (1,'Alice','Hanoi'),
  (2,'Bob','HCMC'),
  (3,'Charlie','Da Nang')
ON CONFLICT (id) DO NOTHING;

INSERT INTO orders (id,user_id,amount) VALUES
  (1,1,120.50),
  (2,1,55.00),
  (3,2,240.10),
  (4,3,15.99)
ON CONFLICT (id) DO NOTHING;