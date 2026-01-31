from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

out_path = "./ML_System_Design_Cheat_Sheet_Meta_L6.pdf"

PAGE_SIZE = landscape(letter)
W, H = PAGE_SIZE

margin = 0.33 * inch
gutter = 0.22 * inch
col_count = 2
usable_w = W - 2*margin - gutter
col_w = usable_w / col_count
col_h = H - 2*margin

styles = getSampleStyleSheet()
base = ParagraphStyle(
    "Base",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.8,
    leading=8.9,
    textColor=colors.black,
    spaceAfter=1.5,
)
h1 = ParagraphStyle(
    "H1",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=12.6,
    leading=13.9,
    spaceAfter=5,
)
h2 = ParagraphStyle(
    "H2",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=10.7,
    spaceBefore=3,
    spaceAfter=1.5,
    textColor=colors.HexColor("#0B3A66"),
)
h3 = ParagraphStyle(
    "H3",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=8.6,
    leading=9.9,
    spaceBefore=2.5,
    spaceAfter=1,
)
small = ParagraphStyle(
    "Small",
    parent=base,
    fontSize=7.2,
    leading=8.3,
)

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(margin, H - margin + 4, "ML System Design Cheat Sheet (Meta L6)")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawRightString(W - margin, H - margin + 6, "Keep crisp: clarify -> numbers -> diagram -> tradeoffs -> ops")
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(margin, margin - 12, "Print: landscape, fit-to-page, margins minimum. Use as a memory jogger (not a script).")
    canvas.drawRightString(W - margin, margin - 12, f"Page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(out_path, pagesize=PAGE_SIZE,
                      leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin)

frames = []
for i in range(col_count):
    x = margin + i*(col_w + gutter)
    frames.append(Frame(x, margin, col_w, col_h, showBoundary=0))

doc.addPageTemplates([PageTemplate(id="TwoCol", frames=frames, onPage=header_footer)])

story = []

# PAGE 1
story.append(Paragraph("ML System Design (Meta L6) — 2-page print cheat sheet", h1))
story.append(Paragraph("<b>Goal:</b> avoid forgetting fundamentals. Drive with structure, explicit assumptions, and quantified tradeoffs.", base))

story.append(Paragraph("1) 2-3 minute kickoff (use this every time)", h2))
kick = [
    "- Restate problem as a product: users, surfaces, business value, constraints.",
    "- Ask clarifying questions (scope): ranking vs retrieval vs generation, regions, on-device vs server, privacy policy, traffic, update cadence.",
    "- Define success: <b>primary metric</b> + guardrails (latency, reliability, fairness, safety/integrity, cost).",
    "- Get numbers: QPS, p95/p99 budgets, model size, feature freshness, storage/compute, label delay.",
    "- Propose 2 approaches (baseline + advanced). Pick one to design end-to-end.",
    "- Draw the diagram early: offline pipeline + online request path + monitoring/feedback.",
]
for line in kick:
    story.append(Paragraph(line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("2) Requirements checklist (functional + non-functional)", h2))
req = [
    "<b>Functional:</b> input/output contract (what comes in; what you return: score/rank/label/embedding), business rules, dedupe/filtering, personalization.",
    "<b>Online path:</b> request -> candidate gen -> feature fetch -> model -> postprocess -> response.",
    "<b>Offline path:</b> events/logs -> labels -> training -> validation -> registry -> deploy.",
    "<b>Non-functional:</b> latency (p50/p95/p99), availability/SLO, scalability (peak/region), privacy/security, interpretability, fairness, cost, operability.",
]
for line in req:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("3) Data & labeling (say how you get trustworthy labels)", h2))
data = [
    "<b>Sources:</b> impression/click/conversion events, content metadata, social graph, feedback/reports, human labels.",
    "<b>Label definition:</b> positive/negative? position bias? delayed rewards? session windows? counterfactual needs?",
    "<b>Sampling:</b> handle imbalance; hard negatives; debias by exposure (IPS) if needed; avoid train/serve skew.",
    "<b>Leakage checks:</b> time-based splits; prevent using future features; consistent joins (as-of).",
    "<b>Quality:</b> bot/spam filtering; dedupe; annotation guidelines + inter-rater agreement; outlier handling.",
    "<b>Freshness:</b> streaming vs batch; late events; backfill strategy; retention/TTL.",
    "<b>Privacy:</b> PII minimization; access controls; encryption; auditing; data retention; differential privacy where appropriate.",
]
for line in data:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("4) Feature engineering & feature store (production parity)", h2))
feat = [
    "<b>Types:</b> user/item/context, cross features, sequence/session, aggregates (windowed counts), embeddings.",
    "<b>Offline/online parity:</b> same transforms; avoid training-only joins; unit tests for parity.",
    "<b>Store design:</b> offline warehouse + online KV + registry/metadata; TTL; versioning; ownership.",
    "<b>Fresh features:</b> idempotent updates; watermarking; time-travel joins; null/default handling.",
    "<b>Feature monitoring:</b> missing rate, distribution drift, cardinality spikes, staleness, schema changes.",
]
for line in feat:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("5) Modeling choices (start simple, then add sophistication)", h2))
model = [
    "<b>Baseline:</b> logistic regression / GBDT / shallow DNN with robust features.",
    "<b>Common shape:</b> multi-stage retrieval + ranking: (retrieve -> rank -> rerank).",
    "<b>Sequence:</b> transformer/RNN for sessions; recency decay; last-N interactions.",
    "<b>Multi-task:</b> shared trunk + heads; label hierarchy; loss weighting.",
    "<b>Constraints:</b> model size vs latency; quantization/distillation; calibration; interpretability needs.",
]
for line in model:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 6))
story.append(Paragraph("Diagram skeleton to draw fast (say it while drawing)", h3))
story.append(Paragraph(
    "Offline: logs/events -> ETL/joins/dedupe -> labeler -> train/validate -> model registry -> deploy. "
    "Online: request -> auth -> candidate gen -> feature fetch -> model inference -> postprocess (business rules, calibration, diversity) -> response. "
    "Sidecars: caches, feature store, experiment config, monitoring/logging, rollback.",
    small
))

# COLUMN 2
story.append(Paragraph("6) Training pipeline (show maturity: reproducibility + scale)", h2))
train = [
    "<b>Data pipeline:</b> snapshot, join, dedupe, time split (avoid leakage), train/val/test; lineage.",
    "<b>Loss & objectives:</b> logloss, pairwise/listwise, sampled softmax, contrastive; regularization; class weights.",
    "<b>Distributed:</b> data/model parallel; mixed precision; checkpointing; sharding; handle stragglers.",
    "<b>Repro:</b> dataset version, code hash, feature schema, seeds, configs; deterministic eval.",
]
for line in train:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("7) Evaluation & validation (offline + online + bias awareness)", h2))
evalv = [
    "<b>Offline metrics:</b> AUC/PR-AUC, logloss, NDCG@k, MRR, Recall@k, calibration (ECE), RMSE.",
    "<b>Slices:</b> device, locale, new vs heavy users, cold-start, content types; long-tail performance.",
    "<b>Biases:</b> position/exposure/selection bias; survivorship; use IPS/counterfactual methods if needed.",
    "<b>Stress tests:</b> adversarial/spam, missing features, distribution shift, extreme load, latency spikes.",
    "<b>Fairness:</b> disparity metrics where applicable; harm analysis; policy constraints.",
]
for line in evalv:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("8) Serving architecture (online inference)", h2))
serve = [
    "<b>Request path:</b> API -> candidate gen -> feature fetch -> model -> postprocess -> response.",
    "<b>Feature fetch:</b> parallel reads; local cache; timeouts; fallback defaults; batching.",
    "<b>Inference:</b> CPU vs GPU; micro-batching; model caching; warmup; thread pools.",
    "<b>Tail latency:</b> hedged requests, circuit breakers, partial results, early exit, load shedding.",
    "<b>Consistency:</b> model/feature version pinning; sticky routing during rollout; schema compatibility.",
]
for line in serve:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("9) Scaling & performance knobs", h2))
scale = [
    "<b>Throughput:</b> batching, vectorization, async I/O, reduce fanout, approximate retrieval.",
    "<b>Caching:</b> embeddings/features/top-K results; invalidation rules; TTL; stampede control.",
    "<b>Sharding:</b> consistent hashing; hot-key mitigation; regionalization; multi-tenant isolation.",
    "<b>Cost:</b> right-size hardware; spot for training; quantize/distill; autoscale by QPS/latency.",
]
for line in scale:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("10) Experimentation, rollout, and monitoring (production muscle)", h2))
exp = [
    "<b>Rollout ladder:</b> offline -> shadow -> canary -> 1% -> 10% -> 50% -> 100% (with holdout).",
    "<b>Metrics:</b> primary + guardrails + long-term; define stop/rollback conditions.",
    "<b>Stat rigor:</b> power, multiple testing, novelty/SRM checks; log configs; backtest.",
    "<b>Monitoring:</b> latency/errors/timeouts; feature missing/staleness; score distributions; drift.",
    "<b>Alerts & debug:</b> actionable thresholds; request tracing; privacy-safe feature dumps; replay.",
]
for line in exp:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("11) Retraining, drift, and continuous improvement", h2))
drift = [
    "<b>Cadence:</b> periodic + triggered retrains (drift, new content, seasonality).",
    "<b>Drift detection:</b> PSI/KS on features; embedding drift; label delay handling; proxies.",
    "<b>Feedback loops:</b> mitigate exploitation/filter bubbles; exploration (epsilon/Thompson).",
    "<b>Human-in-loop:</b> active learning, triage queues, label audits, abuse review.",
]
for line in drift:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 4))
story.append(Paragraph("12) Reliability & safety failure modes (name them explicitly)", h2))
rel = [
    "<b>Cold start:</b> priors + similarity + onboarding; fast embedding init.",
    "<b>Spam/abuse:</b> adversarial content/bots; rate limit; anomaly detection; robust features.",
    "<b>Data outage:</b> stale/missing features -> cached/fallback model; degrade gracefully.",
    "<b>Model regression:</b> canary/shadow/diff tests; automatic rollback; audit trails.",
    "<b>Privacy incidents:</b> PII in features/logs -> minimization; redaction; incident playbook.",
]
for line in rel:
    story.append(Paragraph("- " + line, base))

story.append(Spacer(1, 6))
story.append(Paragraph("L6 signal checklist (what they are listening for)", h2))
signals = [
    "<b>Leadership:</b> you drive ambiguity to clarity; propose plan A/plan B; align on success criteria early.",
    "<b>Tradeoffs:</b> you quantify and choose (latency vs quality vs cost vs freshness); call out risks + mitigations.",
    "<b>End-to-end ownership:</b> data -> model -> serving -> experiments -> monitoring -> incident response.",
    "<b>Pragmatism:</b> pick a baseline, ship safely, iterate; avoid over-engineering.",
    "<b>Depth on demand:</b> zoom in (loss/caching/drift) or zoom out (product/system) smoothly.",
]
for line in signals:
    story.append(Paragraph("- " + line, base))


doc.build(story)

out_path

