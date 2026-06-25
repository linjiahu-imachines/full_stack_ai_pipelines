# PostgreSQL and vector DB setup & verification

Record of commerce database (PostgreSQL) and semantic RAG index (Chroma) for the voice agent on **imu-thor**.

**Host:** Ubuntu 24.04 · PostgreSQL 16 · Chroma persistent store  
**Last updated:** June 2026

---

## Overview

The agent uses **three** knowledge-related stores:

| Store | Technology | Purpose |
|-------|------------|---------|
| **Commerce DB** | PostgreSQL (`horizon_store`) | Structured customer profiles, orders, shipments, products, policies |
| **Vector DB** | Chroma (`server/data/chroma/`) | Semantic search over SQL summaries + knowledge text |
| **BM25 index** | JSON file (`server/data/knowledge_index.json`) | Keyword search; combined with Chroma in `hybrid` mode |

Voice **sessions** remain file-based under `server/data/sessions/` (not in Postgres).

Config block: [`demo_staged_kokoro_agent_dual.yaml`](../project/configs/demo_staged_kokoro_agent_dual.yaml) → `agent.database`

---

## Prerequisites

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
pip install -e ".[vector]"
```

Dependencies: `sqlalchemy`, `psycopg[binary]`, `chromadb`, `sentence-transformers`

---

## Part 1 — PostgreSQL (native install, no Docker)

We use **Option B**: PostgreSQL installed directly on the Thor machine.

### 1.1 Install PostgreSQL

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

Verify:

```bash
systemctl is-active postgresql   # expect: active
psql --version                   # expect: PostgreSQL 16.x
```

### 1.2 Create database and user

Run as the system `postgres` user:

```bash
sudo -u postgres psql <<'EOF'
CREATE USER horizon WITH PASSWORD 'horizon';
CREATE DATABASE horizon_store OWNER horizon;
GRANT ALL PRIVILEGES ON DATABASE horizon_store TO horizon;
EOF
```

- **Username:** `horizon`
- **Password:** set in the `CREATE USER ... PASSWORD '...'` line (demo value: `horizon`)
- **Database:** `horizon_store`

Change password later:

```bash
sudo -u postgres psql -c "ALTER USER horizon WITH PASSWORD 'your_new_password';"
```

### 1.3 Connection string

Format:

```
postgresql+psycopg://USERNAME:PASSWORD@HOST:PORT/DATABASE
```

Local example (used on imu-thor):

```
postgresql+psycopg://horizon:horizon@127.0.0.1:5432/horizon_store
```

Add to `server/.env.local` (loaded automatically at server startup):

```bash
cp server/.env.local.example server/.env.local
nano server/.env.local
```

```ini
DATABASE_SQL_URL=postgresql+psycopg://horizon:horizon@127.0.0.1:5432/horizon_store
```

### 1.4 `pg_hba.conf` (usually not needed)

If this works, skip editing `pg_hba.conf`:

```bash
psql "postgresql://horizon:horizon@127.0.0.1:5432/horizon_store" -c "SELECT 1;"
```

Only if you get authentication errors, check `/etc/postgresql/16/main/pg_hba.conf` for a line like:

```
host    horizon_store    horizon    127.0.0.1/32    scram-sha-256
```

Then: `sudo systemctl restart postgresql`

### 1.5 Initialize schema and seed data

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
python scripts/init_databases.py
```

This creates tables and seeds the **Alex Morgan / Horizon Store** demo data.  
The voice server also auto-seeds if the `customers` table is empty on first startup.

### 1.6 PostgreSQL schema

| Table | Contents |
|-------|----------|
| `customers` | Profile, loyalty, addresses, store credit |
| `orders` | Order headers (`is_active` marks current order) |
| `line_items` | Products per order |
| `shipments` | Carrier, tracking, ETA, delivery |
| `tracking_events` | Shipment timeline |
| `products` | SKU catalog snippets |
| `policies` | Shipping, return, cancellation, etc. |

Agent tools: `lookup_customer`, `lookup_order`, `get_active_order`, `list_customer_orders`

---

## Part 2 — Chroma vector DB

### 2.1 Location and config

| Setting | Value |
|---------|--------|
| Path | `server/data/chroma/` |
| Collection | `horizon_kb` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| RAG mode | `hybrid` (Chroma + BM25) |

YAML (`agent.database`):

```yaml
database:
  enabled: true
  sql_url: postgresql+psycopg://horizon:horizon@127.0.0.1:5432/horizon_store
  vector_path: data/chroma
  vector_collection: horizon_kb
  embed_model: sentence-transformers/all-MiniLM-L6-v2
  rag_backend: hybrid
```

Environment overrides: `DATABASE_SQL_URL`, `DATABASE_VECTOR_PATH`, `DATABASE_VECTOR_COLLECTION`, `DATABASE_EMBED_MODEL`, `DATABASE_RAG_BACKEND`, `DATABASE_ENABLED`

### 2.2 Build or rebuild vector index

First time or after corpus / SQL seed changes:

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
python scripts/init_databases.py --force-vector
```

SQL-only (skip Chroma):

```bash
python scripts/init_databases.py --sql-only
```

Index contents (~74 documents on imu-thor):

- Text summaries exported from PostgreSQL (`customer:…`, `order:…`, `product:…`, `policy:…`)
- Chunks from `server/data/knowledge/alex_ecommerce_rag_context.txt`

---

## Part 3 — Verification (completed on imu-thor)

### 3.1 PostgreSQL — connect

```bash
psql "postgresql://horizon:horizon@127.0.0.1:5432/horizon_store"
```

Inside `psql`:

```sql
\dt

SELECT id, preferred_name, full_name, email, loyalty_tier, loyalty_points
FROM customers;

SELECT id, order_status, order_total, is_active, order_date
FROM orders
ORDER BY is_active DESC, order_date DESC;
```

**Verified row counts:**

| Table | Rows |
|-------|------|
| `customers` | 1 |
| `orders` | 4 |
| `line_items` | 4 |
| `shipments` | 2 |
| `tracking_events` | 2 |
| `products` | 3 |
| `policies` | 4 |

Quit: `\q`

### 3.2 Chroma — document count

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate

python3 -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma')
col = client.get_collection('horizon_kb')
print('Documents:', col.count())
print('Sample IDs:', col.peek(3)['ids'])
"
```

**Verified output:**

```
Documents: 74
Sample IDs: ['customer:CUST-100482', 'order:ORD-2026-0612-7842', 'order:ORD-2026-0203-5119']
```

### 3.3 Chroma — semantic search

```bash
python3 -c "
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path='data/chroma')
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
col = client.get_collection('horizon_kb', embedding_function=ef)
r = col.query(query_texts=['Where is my laptop shipment?'], n_results=3)
for i, doc_id in enumerate(r['ids'][0], 1):
    print(i, doc_id)
"
```

**Expected top hit:** `order:ORD-2026-0612-7842` (active order with NovaBook laptop / UPS shipment)

### 3.4 Voice server health

Start or restart the server, then:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

**Verified agent fields:**

```json
"agent": {
  "enabled": true,
  "database_enabled": true,
  "rag_backend": "hybrid",
  "vector_doc_count": 74,
  "knowledge_chunks": 136,
  "web_search_enabled": true
}
```

---

## Part 4 — Useful SQL queries

**Active order with line items:**

```sql
SELECT o.id AS order_id, o.order_status, li.product_name, li.fulfillment_status, li.shipment_id
FROM orders o
JOIN line_items li ON li.order_id = o.id
WHERE o.is_active = true;
```

**Shipments and tracking:**

```sql
SELECT s.id, s.carrier, s.status, s.tracking_number, s.estimated_delivery_date,
       te.event_time, te.location, te.status AS event_status, te.description
FROM shipments s
LEFT JOIN tracking_events te ON te.shipment_id = s.id
ORDER BY s.id, te.event_time;
```

**Policies:**

```sql
SELECT policy_key, title, body FROM policies;
```

---

## Part 5 — Troubleshooting

| Symptom | Fix |
|---------|-----|
| `connection refused` on port 5432 | `sudo systemctl start postgresql` |
| `password authentication failed` | Check password in `DATABASE_SQL_URL`; `ALTER USER horizon WITH PASSWORD '...'` |
| `Documents: 0` in Chroma | `python scripts/init_databases.py --force-vector` |
| `chromadb not installed` | `pip install -e ".[vector]"` |
| `database_enabled: false` in `/health` | Set `DATABASE_SQL_URL` in `.env.local`; restart uvicorn |
| `vector_doc_count: 0` | Rebuild vector index; confirm `data/chroma/` exists |

---

## Part 6 — Related files

| Path | Role |
|------|------|
| `server/app/db/models.py` | SQLAlchemy schema |
| `server/app/db/seed_data.py` | Multi-customer demo seed (34 use cases) |
| `server/app/db/repositories.py` | Commerce lookups + vector document export |
| `server/app/db/vector_store.py` | Chroma wrapper |
| `server/app/db/hybrid_rag.py` | Hybrid BM25 + vector search |
| `server/scripts/init_databases.py` | One-shot init / rebuild |
| `server/docker-compose.yml` | Optional Docker Postgres (not used on imu-thor) |
| `server/data/knowledge/alex_ecommerce_rag_context.txt` | Original Alex Morgan demo corpus |
| `server/data/knowledge/cs_use_cases_kb.txt` | Customer-service use-case KB (products, policies, troubleshooting) |

---

## Related documents

| Document | Topic |
|----------|--------|
| [CUSTOMER_SERVICE_USE_CASES.md](CUSTOMER_SERVICE_USE_CASES.md) | 34 use cases → Postgres + RAG + tools |
| [RUN_VOICE_PIPELINE.md](RUN_VOICE_PIPELINE.md) | Start/stop voice server |
| [DEMO_VOICE_QUERIES.md](DEMO_VOICE_QUERIES.md) | Demo voice script |
| [SERVER_AND_PROJECT1_ARCHITECTURE.md](SERVER_AND_PROJECT1_ARCHITECTURE.md) | Agent, RAG, tools |
