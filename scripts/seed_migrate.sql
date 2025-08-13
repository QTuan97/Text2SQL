INSERT INTO semantic_synonyms(phrase, maps_to, weight, approved) VALUES
  ('turnover',      'metrics.revenue',      1.0, TRUE),
  ('gmv',           'metrics.revenue',      1.0, TRUE),
  ('order volume',  'metrics.orders_count', 1.0, TRUE)
ON CONFLICT (phrase) DO UPDATE
  SET maps_to=EXCLUDED.maps_to, weight=EXCLUDED.weight, approved=EXCLUDED.approved;

INSERT INTO semantic_metrics(name, expression, unit, synonyms) VALUES
  ('revenue',      'SUM(orders.amount)', 'currency', ARRAY['sales','turnover']),
  ('orders_count', 'COUNT(orders.id)',   NULL,        ARRAY['number of orders','order volume'])
ON CONFLICT (name) DO NOTHING;