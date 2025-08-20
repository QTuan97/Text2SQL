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
      -d '{"limit":100,"with_payload":true,"with_vectors":false,"filter":{"must":[{"key":"kind","match":{"value":"lesson"}},{"key":"question","match":{"value":"Top 5 users from HCMC sort by revenue"}}]}}' \
    | jq '.result.points | map({id:.id,question:.payload.question,sql:.payload.sql,quality:.payload.quality,tables:.payload.tables,created_at:.payload.created_at})'

learn_logs1:
    curl -s -X POST "http://localhost:6333/collections/sql_lessons/points/scroll" \
      -H 'content-type: application/json' \
      -d '{"limit":100,"with_payload":true,"with_vectors":false,"filter":{"must":[{"key":"kind","match":{"value":"lesson"}},{"key":"question","match":{"value":"Top 5 users by revenue"}}]}}' \
    | jq '.result.points | map({id:.id,question:.payload.question,sql:.payload.sql,quality:.payload.quality,tables:.payload.tables,created_at:.payload.created_at})'

learn_logs2:
    curl -s -X POST "http://localhost:6333/collections/sql_lessons/points/scroll" \
      -H 'content-type: application/json' \
      -d '{"limit":100,"with_payload":true,"with_vectors":false,"filter":{"must":[{"key":"kind","match":{"value":"lesson"}},{"key":"question","match":{"value":"Top 5 users sort by total order amount"}}]}}' \
    | jq '.result.points | map({id:.id,question:.payload.question,sql:.payload.sql,quality:.payload.quality,tables:.payload.tables,created_at:.payload.created_at})'

# ── Qdrant env (override when running: QDRANT_URL=... LESSONS_COLLECTION=...) ──
QDRANT_URL           := "http://localhost:6333"
LESSONS_COLLECTION   := "sql_lessons"
VALID_NAME           := "valid_vec"
VALID_DIM            := "768"
ERROR_NAME           := "error_vec"
ERROR_DIM            := "384"

# Drop & recreate the lessons collection with named vectors
qdrant-lessons-reset:
    set -euo pipefail
    if [ -n "${QDRANT_API_KEY:-}" ]; then HDR="-H"; HDRV="api-key: ${QDRANT_API_KEY}"; else HDR=""; HDRV=""; fi
    echo "→ Deleting collection ${LESSONS_COLLECTION} (if exists)…"
    curl -sfS ${HDR-} ${HDRV-} -X DELETE "${QDRANT_URL}/collections/${LESSONS_COLLECTION}" || true
    echo "→ Recreating collection ${LESSONS_COLLECTION} with named vectors…"
    DATA="{\"vectors\":{\"${VALID_NAME}\":{\"size\":${VALID_DIM},\"distance\":\"Cosine\"},\"${ERROR_NAME}\":{\"size\":${ERROR_DIM},\"distance\":\"Cosine\"}}}"
    curl -sfS ${HDR-} ${HDRV-} -X PUT "${QDRANT_URL}/collections/${LESSONS_COLLECTION}" \
      -H 'Content-Type: application/json' \
      -d "$DATA"
    echo "✓ Done."

# Delete all points but keep the collection config
qdrant-lessons-clear:
    set -euo pipefail
    if [ -n "${QDRANT_API_KEY:-}" ]; then HDR="-H"; HDRV="api-key: ${QDRANT_API_KEY}"; else HDR=""; HDRV=""; fi
    echo "→ Clearing all points from ${LESSONS_COLLECTION}…"
    curl -sfS ${HDR-} ${HDRV-} -X POST "${QDRANT_URL}/collections/${LESSONS_COLLECTION}/points/delete?wait=true" \
      -H 'Content-Type: application/json' \
      -d '{"filter":{"must":[]}}'
    echo "✓ Cleared."

# Count points in lessons
qdrant-lessons-count:
    set -euo pipefail
    if [ -n "${QDRANT_API_KEY:-}" ]; then HDR="-H"; HDRV="api-key: ${QDRANT_API_KEY}"; else HDR=""; HDRV=""; fi
    curl -sfS ${HDR-} ${HDRV-} -X POST "${QDRANT_URL}/collections/${LESSONS_COLLECTION}/points/count" \
      -H 'Content-Type: application/json' \
      -d '{"exact": true}' | jq -r '.result.count'