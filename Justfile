compose := "docker compose"
project := "Text2SQL"           # optional
compose_p := "{{compose}} -p {{project}}"
set shell := ["bash","-eu","-o","pipefail","-c"]
set dotenv-load := true
set export := true
DB_SERVICE := "postgres"
DB_USER    := "app"
DB_NAME    := "appdb"
API_URL    := "http://localhost:8000"

QDRANT := "http://localhost:6333"
COL := "sql_lessons"
Q := 'Top 10 users by revenue'

up:
	docker compose up -d --build

down:
	docker compose down

wipe:
    docker compose down -v

logs:
	docker compose logs -f api

health:
	curl -s http://localhost:8000/health

restart:
    just down
    just up

reup:
	just wipe
	just up
	just health

semantic-reload:
	( curl -fsS -X POST "{{API_URL}}/semantic/reload" \
	|| curl -fsS -X POST "{{API_URL}}/api/semantic/reload" ) || true

db-reset:
	test -f scripts/db_reset.sql
	{{compose}} exec -T {{DB_SERVICE}} psql -U {{DB_USER}} -d {{DB_NAME}} -v ON_ERROR_STOP=1 -f - < scripts/db_reset.sql

db-migrate:
	test -f scripts/db_migrate.sql
	{{compose}} exec -T {{DB_SERVICE}} psql -U {{DB_USER}} -d {{DB_NAME}} -v ON_ERROR_STOP=1 -f - < scripts/db_migrate.sql

db-seed:
	test -f scripts/db_seed.sql
	{{compose}} exec -T {{DB_SERVICE}} psql -U {{DB_USER}} -d {{DB_NAME}} -v ON_ERROR_STOP=1 -f - < scripts/db_seed.sql

user-table-migrate:
	{{compose}} exec -T {{DB_SERVICE}} psql -U {{DB_USER}} -d {{DB_NAME}} -v ON_ERROR_STOP=1 -f - < scripts/user_schema.sql

tables:
	{{compose}} exec -T {{DB_SERVICE}} psql -U {{DB_USER}} -d {{DB_NAME}} -c "\dt+ public.*"

qdrant_tables:
    curl -s "http://localhost:8000/schema/tables" | jq .
    curl -s "http://localhost:8000/schema/text" | jq

learn_logs:
    curl -s -X POST "http://localhost:6333/collections/sql_lessons/points/scroll" \
      -H 'content-type: application/json' \
      -d '{"limit":100,"with_payload":true,"with_vectors":false,"filter":{"must":[{"key":"kind","match":{"value":"lesson"}},{"key":"question","match":{"value":"Top 10 users by revenue"}}]}}' \
    | jq '.result.points | map({id:.id,question:.payload.question,sql:.payload.sql,quality:.payload.quality,tables:.payload.tables,created_at:.payload.created_at})'

# Tail Ollama logs
ollama-logs:
  {{compose}} logs -f --tail=200 ollama