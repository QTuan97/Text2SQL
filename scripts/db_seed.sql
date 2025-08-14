-- Deterministic-ish randomness for repeatable tests
SELECT setseed(0.42);

-- Seed 50 users across 5 cities
INSERT INTO users (name, city)
SELECT 'User ' || i AS name,
       (ARRAY['Hanoi','HCMC','Da Nang','Hue','Can Tho'])[1 + (random()*4)::int] AS city
FROM generate_series(1, 50) AS s(i);

-- Seed ~1,000 orders over the last 90 days with varied amounts
INSERT INTO orders (user_id, amount, created_at)
SELECT 1 + (random()*49)::int AS user_id,                          -- users 1..50
       round((10 + random()*490)::numeric, 2) AS amount,           -- 10.00 .. 500.00
       NOW()
         - ((random()*90)::int) * INTERVAL '1 day'
         - ((random()*24)::int) * INTERVAL '1 hour'
FROM generate_series(1, 1000);

-- Helpful semantic overrides to power the hybrid MDL
INSERT INTO semantic_metrics(name, expression, unit, synonyms) VALUES
  ('revenue',      'SUM(orders.amount)', 'currency', ARRAY['sales','turnover','gmv']),
  ('orders_count', 'COUNT(orders.id)',   NULL,        ARRAY['number of orders','order volume'])
ON CONFLICT (name) DO UPDATE
  SET expression = EXCLUDED.expression,
      unit       = EXCLUDED.unit,
      synonyms   = EXCLUDED.synonyms;

INSERT INTO semantic_synonyms(phrase, maps_to, weight, approved) VALUES
  ('turnover',          'metrics.revenue',      1.0, TRUE),
  ('gmv',               'metrics.revenue',      1.0, TRUE),
  ('order volume',      'metrics.orders_count', 1.0, TRUE),
  ('top customers',     'users.name',           1.0, TRUE),
  ('customer city',     'users.city',           1.0, TRUE),
  ('sales by city',     'metrics.revenue',      1.0, TRUE),
  ('sales last month',  'metrics.revenue',      1.0, TRUE)
ON CONFLICT (phrase) DO UPDATE
  SET maps_to  = EXCLUDED.maps_to,
      weight   = EXCLUDED.weight,
      approved = EXCLUDED.approved;