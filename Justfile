compose := "docker compose"
project := "Text2SQL"           # optional
compose_p := "{{compose}} -p {{project}}"
set shell := ["bash","-eu","-o","pipefail","-c"]
set dotenv-load := true
set export := true
API_URL    := "http://localhost:8000"

up:
	docker compose up -d --build

down:
	docker compose down

wipe:
    docker compose down -v

logs:
	docker compose logs -f api

ps:
    docker compose ps

health:
	curl -s http://localhost:8000/health

restart:
    just down
    just up

reup:
	just wipe
	just up
	just health

pg_init:
    docker compose exec -T postgres bash -lc 'psql -U postgres -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname=''test''" | grep -q 1 || psql -U postgres -d postgres -c "CREATE ROLE test LOGIN PASSWORD ''123123'';"'
    docker compose exec -T postgres bash -lc 'psql -U postgres -d postgres -tc "SELECT 1 FROM pg_database WHERE datname=''appdb''" | grep -q 1 || psql -U postgres -d postgres -c "CREATE DATABASE appdb OWNER test;"'
    docker compose exec -T postgres psql -U postgres -d appdb -c "ALTER SCHEMA public OWNER TO app;"

seed_pg:
    docker compose exec -T postgres psql -U app -d appdb < scripts/seed_pg.sql

semantic-migrate:
	docker compose exec -T postgres psql -U app -d appdb -v ON_ERROR_STOP=1 -f - < scripts/migrate.sql

semantic-seed:
	docker compose exec -T postgres psql -U app -d appdb -v ON_ERROR_STOP=1 -f - < scripts/seed_migrate.sql

semantic-reload:
	curl -fsS -X POST "{{API_URL}}/semantic/reload" || true

# Tail Ollama logs
ollama-logs:
  {{compose}} logs -f --tail=200 ollama

# Health & restarts
ollama-health:
  cid=$({{compose}} ps -q ollama); docker inspect -f '{{"{{"}}.State.Status{{"}}"}}
  {{"{{"}}if .State.Health{{"}}"}} {{"{{"}}.State.Health.Status{{"}}"}} {{"{{"}}end{{"}}"}} restarts={{"{{"}}.RestartCount{{"}}"}}' "$$cid"

# Quick API checks
ollama-ping:
  {{compose}} exec -T ollama curl -fsS localhost:11434/api/version || true
  {{compose}} exec -T ollama curl -fsS localhost:11434/api/tags || true