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

# Tail Ollama logs
ollama-logs:
  {{compose}} logs -f --tail=200 ollama