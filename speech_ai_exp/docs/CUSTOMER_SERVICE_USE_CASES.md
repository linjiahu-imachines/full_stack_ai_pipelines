# Customer service use cases — data and tool mapping

Maps the 34 voice-agent use cases to **PostgreSQL**, **Chroma vector RAG**, and **agent tools**.

**Demo customers (PostgreSQL):**

| Customer | ID | Email | Primary use cases |
|----------|-----|-------|-------------------|
| John Smith | `C-001257` | `j.smith@acme.com` | 1–3, 8–9, 16–18, 25–26, 27–28 |
| Sarah Johnson | `C-SJ-2201` | `sarah.johnson@example.test` | 4, 6–7, 24 |
| Chris Taylor | `C-19382` | `chris.taylor@example.test` | 5 |
| Nguyen Thi Minh | `C-AX921` | `minh.nguyen@example.test` | 29, 28 |
| Alex Morgan | `CUST-100482` | `alex.morgan@example.test` | Original Horizon Store demo |

**Key order numbers:** `784921`, `785421`, `918273`, `123456789`

After schema changes:

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
python scripts/init_databases.py --reset-sql --force-vector
```

---

## Use case reference

| ID | Use case | Sample prompt | DB / RAG | Tools / actions |
|----|----------|---------------|----------|-----------------|
| 1 | Order lookup | John Smith, order **784921** | Postgres `ORD-784921` | `lookup_order`, `lookup_customer` |
| 2 | Invoice lookup | Customer **C001257**, latest invoice | Postgres `INV-2026-784921` | `lookup_invoice` (ID normalized to `C-001257`) |
| 3 | Payment status | **j dot smith at acme dot com** | Postgres `payments` | `lookup_payment_status` (email normalized) |
| 4 | Multi-turn context | Sarah Johnson → open orders? | Postgres open orders | Turn 1: `lookup_customer` + `remember`; turn 2: `list_open_orders` |
| 5 | Customer verification | Account info → **C-1-9-3-8-2** | Postgres `C-19382` | Ask for ID → `lookup_customer` |
| 6 | Order tracking | Order **918273** ETA? | Postgres `ORD-918273` + shipment | `lookup_order` |
| 7 | Delayed orders | Which orders delayed? | Postgres `is_delayed=true` | `list_delayed_orders` (Sarah) |
| 8 | Return policy | Laptop last week, return? | RAG + John’s laptop line item | `lookup_order` + `search_knowledge_base` |
| 9 | Refund eligibility | Opened package, unused | RAG refund policy | `search_knowledge_base` |
| 10 | Product features | Model **X500** Wi-Fi 7? | RAG + product `RT-X500` | `search_knowledge_base` |
| 11 | Plan comparison | Premium vs enterprise | Postgres `support_plans` + RAG | `compare_support_plans`, `search_knowledge_base` |
| 12 | Compatibility | Printers + macOS Ventura | RAG `cs_use_cases_kb.txt` | `search_knowledge_base` |
| 13 | Troubleshooting | Router reboot loop | RAG X500 troubleshooting | `search_knowledge_base` |
| 14 | Error code | Printer **E57** | RAG + KB article `kb-e57` | `search_knowledge_base` |
| 15 | Login issue | Can’t log in after update | RAG login KB | `search_knowledge_base` |
| 16 | Warranty claim | Server two weeks ago | Postgres `ORD-784500` + RAG warranty | `lookup_order` + `search_knowledge_base` |
| 17 | Invoice explanation | Invoice charges wrong | Postgres invoice line summary + RAG | `lookup_invoice` + `search_knowledge_base` |
| 18 | License status | Three licenses, how many active? | Postgres licenses (2 active / 3) | `lookup_licenses` (John) |
| 19 | Security requirements | Government deployment security | RAG government security section | `search_knowledge_base` |
| 20 | Policy comparison | DR vs HA policy | RAG + `kb-dr-ha` | `search_knowledge_base` |
| 21 | GDPR | GDPR retention | RAG + `kb-gdpr` | `search_knowledge_base` |
| 22 | Ambiguous order | Problem with my order | — | Ask order # or name → `lookup_customer` |
| 23 | Broken product | Thing from last month broken | — | Clarify product → `list_customer_orders` |
| 24 | Order optimization | 3 open orders, $10k budget | Sarah 3 open orders + penalties | `list_open_orders` + reasoning |
| 25 | License planning | 5 licenses, $500 budget | RAG license pricing | `search_knowledge_base` |
| 26 | Plan recommendation | Support plan from history | John Premium subscription | `lookup_subscription` + `compare_support_plans` |
| 27 | Speech self-correction | 784… no 785421 | Orders `784921` and `785421` | `lookup_order` with corrected number |
| 28 | Long numeric sequence | Order **123456789** | Postgres `ORD-123456789` | `lookup_order` |
| 29 | International names | Nguyen Thi Minh, **AX921** | Postgres `C-AX921` | `lookup_customer` |
| 30 | Disfluency | um, ordered printer in February… | — | ASR + `search_knowledge_base` / SQL |
| 31 | Production outage | Server down 3 hours | RAG SLA Severity 1 | `search_knowledge_base` |
| 32 | Human escalation | Speak to human now | RAG escalation workflow | `search_knowledge_base`; acknowledge escalation |
| 33 | Product lifecycle | Products losing support next year | RAG EOL section | `search_knowledge_base` |
| 34 | Subscription renewal | Renewal amount / downgrade | Postgres `subscriptions` | `lookup_subscription` |

---

## PostgreSQL tables

| Table | Use |
|-------|-----|
| `customers` | Profiles, verification |
| `orders` | Status, `display_order_number`, `is_open`, `is_delayed`, penalties |
| `line_items` | Products per order, return windows |
| `shipments` / `tracking_events` | Tracking and ETA |
| `invoices` | Invoice lookup and charge explanation |
| `payments` | Payment captured / failed |
| `licenses` | Active license count |
| `subscriptions` | Renewal and downgrade |
| `support_plans` | Premium vs Enterprise |
| `products` | SKU catalog |
| `policies` | Return, refund, warranty, GDPR |
| `knowledge_articles` | Short KB rows (also in vector index) |

---

## Vector RAG sources

| Source | Path |
|--------|------|
| SQL summaries | Exported on `init_databases.py` |
| Use-case KB | `server/data/knowledge/cs_use_cases_kb.txt` |
| Original demo | `server/data/knowledge/alex_ecommerce_rag_context.txt` |

---

## Agent tools (PostgreSQL-backed)

| Tool | Purpose |
|------|---------|
| `lookup_customer` | By ID, email, or name |
| `lookup_order` | By order ID or spoken number |
| `lookup_invoice` | Latest or specific invoice |
| `lookup_payment_status` | Payment captured? |
| `list_open_orders` | Open / in-progress orders |
| `list_delayed_orders` | Delayed orders |
| `lookup_licenses` | License count and status |
| `lookup_subscription` | Renewal and downgrade |
| `compare_support_plans` | Premium vs Enterprise |
| `get_active_order` | Current active order |
| `list_customer_orders` | Order history |
| `search_knowledge_base` | Hybrid BM25 + vector semantic search |
| `remember` / `recall` | Multi-turn customer context (session) |

---

## Verification commands

```bash
# SQL spot checks
psql "postgresql://horizon:horizon@127.0.0.1:5432/horizon_store" -c \
  "SELECT id, display_order_number, order_status FROM orders;"

psql "postgresql://horizon:horizon@127.0.0.1:5432/horizon_store" -c \
  "SELECT id, email FROM customers;"

# Vector count
python3 -c "
import chromadb
c = chromadb.PersistentClient(path='data/chroma')
print('docs:', c.get_collection('horizon_kb').count())
"
```

---

## Related

- [DATABASE_SETUP.md](DATABASE_SETUP.md) — Postgres + Chroma install
- [DEMO_VOICE_QUERIES.md](DEMO_VOICE_QUERIES.md) — Alex Morgan 4-turn voice demo
