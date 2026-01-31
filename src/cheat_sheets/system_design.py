from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

out_path = "./system_design_cheat_sheet.pdf"

PAGE_SIZE = landscape(letter)
W, H = PAGE_SIZE

margin = 0.35 * inch
gutter = 0.25 * inch
col_count = 2
usable_w = W - 2*margin - gutter
col_w = usable_w / col_count
col_h = H - 2*margin

styles = getSampleStyleSheet()
base = ParagraphStyle(
    "Base",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.2,
    leading=9.4,
    textColor=colors.black,
    spaceAfter=2,
)
h1 = ParagraphStyle(
    "H1",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=12.8,
    leading=14.2,
    spaceAfter=6,
)
h2 = ParagraphStyle(
    "H2",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=9.7,
    leading=11.2,
    spaceBefore=4,
    spaceAfter=2,
    textColor=colors.HexColor("#0B3A66"),
)
h3 = ParagraphStyle(
    "H3",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=8.7,
    leading=10.0,
    spaceBefore=3,
    spaceAfter=1,
)
small = ParagraphStyle(
    "Small",
    parent=base,
    fontSize=7.4,
    leading=8.5,
    spaceAfter=1.5,
)
tiny = ParagraphStyle(
    "Tiny",
    parent=base,
    fontSize=6.9,
    leading=7.8,
    spaceAfter=1.2,
)

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(margin, H - margin + 4, "End-to-End System Design Cheat Sheet (L6 signals)")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawRightString(W - margin, H - margin + 6, "clarify -> numbers -> diagram -> APIs/data -> scaling -> reliability -> security -> cost")
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(margin, margin - 12, "Print: landscape, fit-to-page, margins minimum. Use as a memory jogger (not a script).")
    canvas.drawRightString(W - margin, margin - 12, f"Page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(out_path, pagesize=PAGE_SIZE,
                      leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin)
frames = [Frame(margin + i*(col_w+gutter), margin, col_w, col_h, showBoundary=0)
          for i in range(col_count)]
doc.addPageTemplates([PageTemplate(id="TwoCol", frames=frames, onPage=header_footer)])

def add_bullets(lines, style=base, prefix="- "):
    for ln in lines:
        story.append(Paragraph(prefix + ln, style))

story = []

# PAGE 1
story.append(Paragraph("System Design (without a dedicated round) — how to signal end-to-end strength", h1))
story.append(Paragraph("<b>Use this to sprinkle system design maturity</b> during coding/ML rounds: requirements, API + data model, scaling, reliability, security, ops, cost. Keep tradeoffs explicit.", base))

story.append(Paragraph("1) 2–3 minute kickoff script", h2))
add_bullets([
    "Restate problem as product: users, core workflow, critical paths. Clarify scope + non-goals.",
    "Get numbers: DAU/MAU, read/write QPS, payload size, p95/p99 latency, availability target, data retention.",
    "Define success: primary metric + guardrails (latency, correctness, privacy, abuse, cost).",
    "Pick a baseline architecture; mention 1–2 future upgrades. Draw diagram early (clients → edge → services → data).",
    "Call out risks (top 3) + mitigations (rate limiting, idempotency, backpressure, fallbacks).",
])

story.append(Spacer(1, 4))
story.append(Paragraph("2) Diagram skeleton (draw fast)", h2))
story.append(Paragraph(
    "<b>Clients</b> → <b>Edge</b> (CDN/WAF/rate limit) → <b>API Gateway</b> → "
    "<b>Stateless services</b> (auth, core, fanout, search) → <b>Data</b> (DB/cache/blob) → "
    "<b>Async</b> (queue/stream) → <b>workers</b> (compute, indexing, notifications) → "
    "<b>Observability</b> (logs/metrics/traces) + <b>config/experiments</b> + <b>CI/CD</b>.",
    small
))

story.append(Spacer(1, 4))
story.append(Paragraph("3) APIs & data modeling checklist", h2))
add_bullets([
    "<b>API:</b> endpoints, authZ/authN, pagination, filtering/sorting, idempotency keys for writes, versioning.",
    "<b>Entities:</b> define IDs (UUID/ULID), ownership, relationships, cardinalities.",
    "<b>Access patterns:</b> list the top queries (by key, by user, by time, search). Model around them.",
    "<b>Consistency:</b> what must be strongly consistent vs eventual (e.g., balances vs feeds).",
    "<b>Schema evolution:</b> additive changes, defaults, backfill strategy.",
], base, prefix="• ")

story.append(Spacer(1, 4))
story.append(Paragraph("4) Storage choices with pros/cons", h2))
add_bullets([
    "<b>Relational (Postgres/MySQL):</b> +ACID, joins, constraints, mature tooling; −harder horizontal sharding, careful with hot rows.",
    "<b>Key-Value (Redis/Dynamo-style):</b> +low latency, simple scaling; −limited queries/joins, modeling upfront, eventual consistency variants.",
    "<b>Wide-column (Cassandra/HBase):</b> +high write throughput, predictable access by partition key; −query flexibility, operational complexity, consistency tradeoffs.",
    "<b>Search index (Elasticsearch/OpenSearch):</b> +full-text + filtering; −eventual consistency, tuning/ops cost, reindexing.",
    "<b>Object store (S3/GCS):</b> +cheap durable blobs; −not for low-latency small reads without caching/CDN.",
    "<b>Time-series/log store:</b> +append-heavy metrics/events; −not for OLTP queries.",
], tiny)

story.append(Spacer(1, 4))
story.append(Paragraph("5) Caching & edge: pros/cons + where it fits", h2))
add_bullets([
    "<b>CDN:</b> +offload global reads, low latency; −cache invalidation, personalization limits.",
    "<b>Service cache (Redis/Memcached):</b> +reduce DB load, speed reads; −staleness, stampedes, eviction surprises.",
    "<b>Patterns:</b> cache-aside (simple), read-through (centralized), write-through (fresh but slower), write-behind (fast but risk).",
    "<b>Stampede control:</b> request coalescing, probabilistic early refresh, jittered TTLs, soft TTL + background refresh.",
], tiny)

story.append(Spacer(1, 4))
story.append(Paragraph("6) Scalability mechanics (what you say out loud)", h2))
add_bullets([
    "<b>Scale reads:</b> caching, replicas, denormalized read models, precompute, pagination, avoid N+1 fanout.",
    "<b>Scale writes:</b> sharding/partitioning by tenant/user/time; batch, async pipelines, idempotent retries.",
    "<b>Sharding:</b> consistent hashing; beware hot keys; add random prefix or split heavy users.",
    "<b>Replication:</b> leader/follower (read replicas) vs multi-leader (conflicts) vs quorum.",
    "<b>Backpressure:</b> bounded queues, timeouts, shedding, priority lanes for critical traffic.",
], base)

# COLUMN 2
story.append(Paragraph("7) Reliability patterns + tradeoffs (L6 signals)", h2))
add_bullets([
    "<b>Timeouts + retries:</b> +mask transient failures; −retry storms. Use exponential backoff + jitter + budgets.",
    "<b>Circuit breaker:</b> +protect dependencies; −needs tuning, can cause partial outages if mis-set.",
    "<b>Bulkheads:</b> +isolate tenants/features; −more capacity planning.",
    "<b>Graceful degradation:</b> +keep core up; −reduced feature quality (serve cached, skip expensive steps).",
    "<b>Idempotency:</b> +safe retries; −requires keys + dedupe storage/window.",
    "<b>Exactly-once myth:</b> aim for at-least-once + idempotent handlers; use transactional outbox where needed.",
], tiny)

story.append(Spacer(1, 4))
story.append(Paragraph("8) Queues/streams & async processing (pros/cons)", h2))
add_bullets([
    "<b>Message queue (SQS/RabbitMQ):</b> +simple work distribution, retries/DLQ; −ordering/throughput limits depending on system.",
    "<b>Log/stream (Kafka/Pulsar):</b> +high throughput, replay, multiple consumers; −ops complexity, ordering per partition, schema discipline.",
    "<b>DLQ:</b> +prevents poison-pill blocking; −needs re-drive process + monitoring.",
    "<b>Exactly-once needs:</b> transactional writes or idempotent consumer + dedupe key + offset management.",
], tiny)

story.append(Spacer(1, 4))
story.append(Paragraph("9) Consistency, correctness, and distributed reality", h2))
add_bullets([
    "<b>CAP talk-track:</b> under partition you pick Consistency vs Availability; choose per operation.",
    "<b>Strong consistency:</b> +simpler correctness; −higher latency/less availability.",
    "<b>Eventual consistency:</b> +availability/latency; −stale reads, anomalies; must design UX and reconciliation.",
    "<b>Common tools:</b> version numbers, compare-and-swap, read-repair, conflict resolution (LWW or domain-specific).",
    "<b>Transactions:</b> local transactions preferred; cross-service transactions are hard → sagas/compensation.",
], base)

story.append(Spacer(1, 4))
story.append(Paragraph("10) Security, privacy, and abuse-resilience", h2))
add_bullets([
    "<b>AuthN/AuthZ:</b> JWT/session, service-to-service mTLS; least privilege; audit logs.",
    "<b>PII:</b> minimize collection, encrypt at rest/in transit, retention limits, redaction in logs.",
    "<b>Abuse:</b> rate limiting, bot detection, input validation, WAF rules, anomaly alerts, per-tenant quotas.",
    "<b>Multi-tenancy:</b> isolation boundaries (schema/DB/cluster), noisy neighbor controls.",
], base)

story.append(Spacer(1, 4))
story.append(Paragraph("11) Observability & operations (what strong candidates mention)", h2))
add_bullets([
    "<b>Golden signals:</b> latency, traffic, errors, saturation (per service + dependency).",
    "<b>Tracing:</b> correlation IDs across services; structured logs; sampled payload logging with PII guardrails.",
    "<b>SLIs/SLOs:</b> define + error budget; alert on symptoms not causes.",
    "<b>Runbooks:</b> dashboards, common failure modes, rollback steps, paging thresholds.",
], base)

story.append(Spacer(1, 4))
story.append(Paragraph("12) Deployment & data migrations (pros/cons)", h2))
add_bullets([
    "<b>Blue/green:</b> +fast rollback; −double capacity.",
    "<b>Canary:</b> +safe gradual rollout; −needs good metrics + routing controls.",
    "<b>Feature flags:</b> +decouple deploy from release; −flag debt.",
    "<b>DB migrations:</b> expand/contract; dual writes carefully; backfills throttled; verify before cutover.",
    "<b>Disaster recovery:</b> backups + restore drills; multi-region active-passive vs active-active tradeoffs.",
], tiny)

story.append(Spacer(1, 6))
story.append(Paragraph("How to “sprinkle” this in non-system-design rounds", h2))
add_bullets([
    "When you propose any pipeline, add: <b>API contract</b>, <b>data store</b>, <b>cache</b>, <b>async job</b>, <b>monitoring</b>, <b>rollback</b>.",
    "Use mini tradeoffs: “I’ll start with Postgres + Redis; if QPS grows, shard by user_id and move heavy queries to read model/search index.”",
    "End with ops: “Canary 1%, watch p99 + error rate + business metric; if regression → auto rollback; alert thresholds + dashboard.”",
], base)

doc.build(story)
out_path

