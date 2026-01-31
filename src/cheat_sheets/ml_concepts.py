from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

out_path = "./ML_Concepts_Cheat_Sheet_Pros_Cons_4pages.pdf"

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
    "Base", parent=styles["Normal"],
    fontName="Helvetica", fontSize=8.2, leading=9.4,
    textColor=colors.black, spaceAfter=1.5
)
h1 = ParagraphStyle(
    "H1", parent=base,
    fontName="Helvetica-Bold", fontSize=12.7, leading=14.0,
    spaceAfter=5
)
h2 = ParagraphStyle(
    "H2", parent=base,
    fontName="Helvetica-Bold", fontSize=9.9, leading=11.0,
    spaceBefore=4.0, spaceAfter=2.0,
    textColor=colors.HexColor("#0B3A66"),
)
h3 = ParagraphStyle(
    "H3", parent=base,
    fontName="Helvetica-Bold", fontSize=8.6, leading=10.0,
    spaceBefore=3.0, spaceAfter=1.5,
    textColor=colors.HexColor("#123B5B"),
)
tiny = ParagraphStyle(
    "Tiny", parent=base,
    fontSize=7.7, leading=8.6, spaceAfter=1.1
)
micro = ParagraphStyle(
    "Micro", parent=base,
    fontSize=7.0, leading=8.0, spaceAfter=1.0
)

def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(margin, H - margin + 4, "ML Concepts Cheat Sheet — pros/cons (L6 refresher)")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawRightString(W - margin, H - margin + 6, "use: choose method -> state assumptions -> +/− -> metrics -> pitfalls")
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 7.3)
    canvas.drawString(margin, margin - 12, "Print: landscape, fit-to-page, margins minimum. Dense catalog; not a tutorial.")
    canvas.drawRightString(W - margin, margin - 12, f"Page {doc.page}/2")
    canvas.restoreState()

doc = BaseDocTemplate(out_path, pagesize=PAGE_SIZE,
                      leftMargin=margin, rightMargin=margin,
                      topMargin=margin, bottomMargin=margin)

frames = [Frame(margin + i*(col_w+gutter), margin, col_w, col_h, showBoundary=0)
          for i in range(col_count)]
doc.addPageTemplates([PageTemplate(id="TwoCol", frames=frames, onPage=header_footer)])

story = []

def bullets(items, style=tiny, prefix="• "):
    for it in items:
        story.append(Paragraph(prefix + it, style))

def item(name, pros, cons, style=micro):
    # Format: Name: + ...; − ...
    txt = f"<b>{name}</b>: <b>+</b> {pros} <b>−</b> {cons}"
    story.append(Paragraph(txt, style))

# ---------------- PAGE 1 ----------------
story.append(Paragraph("Machine Learning — compressed “everything” map (with pros/cons)", h1))
story.append(Paragraph(
    "<b>Workflow invariants:</b> define target + data → split correctly → pick baseline → tune/regularize → evaluate (incl. slices) → calibrate/threshold → deploy + monitor drift.",
    base
))

story.append(Paragraph("Problem types & core framing", h2))
bullets([
    "<b>Supervised:</b> regression, classification, ranking. Key: label quality, leakage, imbalance, costs.",
    "<b>Unsupervised:</b> clustering, density estimation, dimensionality reduction. Key: evaluation is indirect.",
    "<b>Self-supervised:</b> learn representations from pretext tasks (contrastive/masked).",
    "<b>Reinforcement learning:</b> sequential decisions, delayed reward; exploration vs exploitation.",
    "<b>Structured prediction:</b> sequences/graphs/sets; needs specialized losses/decoding.",
    "<b>Generative modeling:</b> model data distribution; sampling quality vs likelihood.",
], tiny)

story.append(Paragraph("Data & splitting (common failure modes)", h2))
bullets([
    "<b>Split:</b> random (IID), <b>time-based</b> (avoid future leakage), user/group split (avoid identity leakage).",
    "<b>Leakage:</b> target leakage, time-travel joins, label leakage via features, preprocessing on full data.",
    "<b>Imbalance:</b> use PR-AUC, class weights, focal loss, resampling; calibrate after reweighting.",
    "<b>Missing/outliers:</b> explicit missing indicators; robust losses; winsorize where justified.",
    "<b>Bias/variance:</b> high bias → add features/model capacity; high variance → regularize/more data/ensembles.",
], tiny)

story.append(Paragraph("Classical supervised algorithms (when to use)", h2))
item("Linear regression", "fast, interpretable, convex; great baseline", "underfits nonlinearity; sensitive to outliers (use Huber/RANSAC)")
item("Ridge/Lasso/Elastic Net", "controls overfit; L1 gives sparsity/feature selection", "L1 unstable with correlated features; scaling matters")
item("Logistic regression", "strong baseline; calibrated-ish; handles large sparse", "linear boundary; needs feature engineering")
item("Naive Bayes", "very fast; good for text; small data", "strong independence assumption; weaker accuracy if violated")
item("k-NN", "simple; non-parametric; good local patterns", "slow at inference; curse of dimensionality; needs scaling")
item("Decision tree", "interpretable; handles nonlinearity; mixed types", "high variance; unstable; overfits without pruning")
item("Random forest", "robust; strong default; handles interactions", "larger models; less interpretable; slower than linear")
item("Gradient boosting (XGBoost/LightGBM/CatBoost)", "SOTA on tabular; handles nonlinearity; good with missing", "tuning sensitive; can overfit; slower training; leakage still kills")
item("SVM (linear/kernel)", "strong margin; kernel handles nonlinearity", "kernel doesn’t scale; tuning C/γ; probability calibration needed")
item("Gaussian processes", "uncertainty + flexible kernels; great small data", "O(n³) training; hard at scale; kernel choice critical")

story.append(Paragraph("Unsupervised & representation learning", h2))
item("PCA", "fast linear DR; denoise; interpretable components", "linear only; variance ≠ usefulness; scaling matters")
item("t-SNE/UMAP", "great visualization of manifolds", "not for downstream metrics; unstable; parameters affect layout")
item("k-means", "fast; simple; scalable", "assumes spherical clusters; needs k; sensitive to init/outliers")
item("GMM (EM)", "soft clusters; ellipses; density estimate", "local minima; assumes Gaussian; needs k/model selection")
item("DBSCAN/HDBSCAN", "finds arbitrary shapes; detects noise", "parameter sensitive; varying density is hard (HDBSCAN helps)")
item("Isolation Forest", "strong anomaly baseline; works high-dim", "interpretability limited; contamination assumptions")
item("One-class SVM", "anomaly detection w/ boundary", "scales poorly; parameter sensitive")
item("Autoencoder (AE)", "learn nonlinear embeddings; anomaly via recon error", "may reconstruct anomalies; needs tuning/regularization")

# ---------------- COLUMN 2 (Page 1) ----------------
story.append(Spacer(1, 60))  # ~6 lines of vertical space
story.append(Paragraph("Neural networks: components, architectures, and training knobs", h2))
story.append(Paragraph("Activations (choose for gradient flow + expressivity)", h3))
item("ReLU", "simple; sparse; fast; default for MLP/CNN", "dead neurons; unbounded outputs")
item("Leaky ReLU/PReLU", "mitigates dead ReLU", "extra hyperparams; still unbounded")
item("ELU/SELU", "better mean/variance dynamics", "slower; SELU requires specific init/alpha-dropout")
item("GELU/Swish", "smooth; strong for Transformers", "slightly slower; can be less stable in some setups")
item("Sigmoid", "probabilities; gates", "vanishing gradients; not for deep hidden layers")
item("tanh", "zero-centered vs sigmoid", "still saturates; vanishing gradients")

story.append(Paragraph("Normalization & regularization", h3))
item("BatchNorm", "stabilizes training; allows higher LR", "batch-size dependent; train/serve mismatch; less common in Transformers")
item("LayerNorm", "stable for sequences/Transformers", "extra compute; can hurt some CNNs vs BN")
item("GroupNorm", "works with small batch sizes", "often slower; may underperform BN at large batch")
item("Dropout", "reduces co-adaptation; simple", "can slow convergence; not always helpful with BN; tune rate")
item("Weight decay (L2 / AdamW)", "strong regularizer; simple", "too much hurts fit; separate from LR schedule (AdamW best practice)")
item("Early stopping", "cheap; prevents overfit", "needs reliable val; can stop too early with noise")

story.append(Paragraph("Optimizers & LR schedules", h3))
item("SGD", "good generalization; stable", "needs tuning; slower convergence")
item("SGD+momentum/Nesterov", "faster; standard for vision", "still needs LR schedule tuning")
item("Adam", "fast convergence; good default", "may generalize worse; sensitive to weight decay coupling")
item("AdamW", "fixes weight decay coupling", "still needs warmup/schedule")
item("RMSProp/Adagrad", "handles sparse/ill-conditioned", "Adagrad LR decays too much; RMSProp less common now")
bullets([
    "<b>Schedules:</b> cosine/linear decay, step, one-cycle, warmup (esp. Transformers), ReduceLROnPlateau.",
    "<b>Init:</b> Xavier/Glorot (tanh), He/Kaiming (ReLU), orthogonal for RNNs.",
], tiny)

story.append(Paragraph("Loss functions (what they optimize) — pros/cons", h2))
item("MSE", "smooth; optimizes mean; standard regression", "sensitive to outliers; blurs multimodal targets")
item("MAE", "robust; optimizes median", "non-smooth at 0; slower convergence")
item("Huber", "robust + smooth", "δ threshold tuning")
item("Cross-entropy (softmax)", "standard classification; well-behaved gradients", "needs label noise handling; can be miscalibrated")
item("Binary cross-entropy", "multi-label; logistic", "thresholding/calibration needed")
item("Hinge", "margin-based; SVM-style", "not probabilistic; less common in deep nets")
item("Focal loss", "handles class imbalance; hard examples", "γ tuning; can hurt calibration")
item("Label smoothing", "improves generalization; reduces overconfidence", "hurts if you need true probabilities; tune ε")
item("Contrastive / Triplet", "metric learning; embeddings", "mining negatives/hard pairs required; collapse risk")
item("InfoNCE", "self-supervised contrastive; strong reps", "needs many negatives or large batch/memory bank")
item("Pairwise ranking (hinge/logistic)", "directly optimizes ordering", "sampling bias; needs good negatives")
item("Listwise (softmax/NDCG surrogates)", "closer to ranking metric", "more complex; can be unstable/tuning-heavy")
item("CTC", "alignment-free sequence labeling", "assumes conditional independence; decoding complexity")
item("Quantile / pinball", "predicts conditional quantiles", "needs quantile choice; can be unstable")
item("GAN losses (minimax/hinge/WGAN-GP)", "sharp samples; implicit density", "training instability; mode collapse; sensitive to tricks")
item("Diffusion training (noise pred / score)", "stable; high sample quality", "slow sampling (mitigated by distillation/fast samplers)")
item("VAE ELBO", "likelihood + latent structure", "posterior collapse; blurry samples vs GAN/diffusion")

story.append(Paragraph("Architectures (when they shine)", h2))
item("MLP", "tabular + embeddings; simplest deep baseline", "weak inductive bias for images/sequences")
item("CNN", "translation equivariance; efficient; vision", "global context harder; architecture tuning")
item("RNN/LSTM/GRU", "streaming/low latency seq; small models", "hard long-range deps; slower than attention")
item("Transformer", "long-range deps; parallelizable; SOTA NLP/vision", "O(n²) attention cost; memory heavy; needs data/compute")
item("GNN (GCN/GAT/GraphSAGE)", "relational data; inductive on graphs", "oversmoothing; sampling; deployment complexity")
item("Mixture-of-Experts", "scale capacity w/ sparse compute", "routing instability; systems complexity; load balancing")

story.append(PageBreak())

# ---------------- PAGE 2, COLUMN 1 ----------------
story.append(Paragraph("Evaluation, calibration, uncertainty, interpretability, fairness", h2))

story.append(Paragraph("Metrics (pick to match business objective)", h3))
bullets([
    "<b>Classification:</b> accuracy (only if balanced), precision/recall/F1, ROC-AUC (ranking), PR-AUC (imbalance), logloss (prob quality).",
    "<b>Calibration:</b> ECE/MCE, reliability curves; calibrate with Platt/Isotonic/temperature scaling.",
    "<b>Ranking:</b> NDCG@k, MAP, MRR, Recall@k; report at multiple k; watch position bias.",
    "<b>Regression:</b> RMSE, MAE, R², MAPE/SMAPE (careful near 0).",
    "<b>Vision:</b> mAP, IoU/Dice, top-k acc; <b>NLP:</b> BLEU/ROUGE (limited), exact match; <b>Gen:</b> human eval + safety.",
    "<b>Operational:</b> p95/p99 latency, cost/query, memory, availability, drift metrics.",
], tiny)

story.append(Paragraph("Model selection & tuning", h3))
item("Cross-validation", "better estimate with small data", "expensive; leakage if groups/time ignored")
item("Grid search", "simple; exhaustive for small spaces", "blows up combinatorially")
item("Random search", "efficient in high-dim", "non-adaptive; still expensive")
item("Bayesian optimization", "sample-efficient; adaptive", "overhead; noisy objectives; implementation complexity")
item("Early pruning (ASHA/Hyperband)", "saves compute", "can kill late-blooming configs")

story.append(Paragraph("Imbalanced / noisy labels", h3))
bullets([
    "<b>Imbalance fixes:</b> class weights, focal loss, undersample negatives, oversample positives (careful), hard-negative mining.",
    "<b>Noise:</b> robust losses, label smoothing, bootstrapping, confident learning; audit labeling pipeline.",
    "<b>Thresholding:</b> optimize for cost curve; choose operating point per segment; consider calibration.",
], tiny)

story.append(Paragraph("Uncertainty & ensembling", h3))
item("Deep ensembles", "strong uncertainty + accuracy", "training cost ×N; deployment complexity")
item("MC dropout", "cheap uncertainty approx", "calibration varies; slower inference (multiple passes)")
item("Bayesian methods / Laplace", "principled uncertainty", "hard to scale; approximations")
item("Quantile regression", "prediction intervals", "only for certain losses/tasks")

story.append(Paragraph("Interpretability", h3))
item("Coefficients (linear)", "transparent global explanation", "misses interactions/nonlinearities")
item("Tree feature importance", "fast global signal", "biased toward high-cardinality; unstable")
item("SHAP", "strong local attributions", "compute heavy; assumptions; can be misused")
item("LIME", "model-agnostic local explanations", "unstable; depends on perturbation distribution")
item("Counterfactual explanations", "actionable what-if", "hard constraints; may be unrealistic")

story.append(Paragraph("Fairness / harm reduction", h3))
bullets([
    "<b>Metrics:</b> demographic parity, equalized odds/opportunity, calibration by group; also long-tail user harm.",
    "<b>Mitigations:</b> reweighting, constraints, adversarial debiasing, post-processing thresholds.",
    "<b>Tradeoff:</b> fairness constraints can reduce global metric; define policy + measurement plan.",
], tiny)

story.append(Paragraph("Causal inference (when correlation is not enough)", h2))
item("A/B testing", "gold standard; causal", "needs time/traffic; interference; novelty effects")
item("Uplift modeling", "targets treatment effect heterogeneity", "label is counterfactual; noisy; needs careful eval")
item("Propensity/IPS", "debias observational", "variance; needs good propensity model")
item("Doubly robust", "less bias/variance", "more complexity; still assumptions")

story.append(Paragraph("Common pitfalls (say these to sound senior)", h2))
bullets([
    "Train/serve skew; leakage; wrong split; hidden objective mismatch (opt metric ≠ business).",
    "Over-optimizing offline metrics; ignoring calibration/threshold; ignoring slices/long tail.",
    "Ignoring data quality + label delay; not monitoring drift; no rollback.",
    "Spurious correlations; fairness regressions; feedback loops (recs/ads).",
], tiny)

# ---------------- COLUMN 2 (Page 2) ----------------
story.append(Spacer(1, 70))  # ~6 lines of vertical space
story.append(Paragraph("Modern ML: self-supervised, NLP/CV, generative models, RL, time series", h2))

story.append(Paragraph("Embeddings & similarity search", h3))
item("Dense embeddings", "capture semantics; reuse across tasks", "need good negatives; drift; embedding staleness")
item("Approx NN (HNSW/IVF/PQ)", "fast retrieval at scale", "recall/latency tradeoff; index build/update complexity")
item("Bi-encoder vs cross-encoder", "bi: fast retrieval; cross: accurate rerank", "bi loses interaction; cross expensive")

story.append(Paragraph("Self-supervised learning", h3))
item("Contrastive (SimCLR/MoCo)", "strong representations; label-free", "needs augmentations/negatives; batch/memory heavy")
item("Masked modeling (BERT/MAE)", "works without negatives; scalable", "pretext-task mismatch; compute heavy")
item("Distillation", "smaller/faster models", "teacher bias; needs careful objective")

story.append(Paragraph("NLP / LLM training & adaptation", h3))
item("SFT (supervised fine-tune)", "aligns to task; simple", "needs quality data; overfitting/catastrophic forgetting")
item("LoRA/PEFT", "cheap adaptation; small deltas", "still needs eval; may not match full fine-tune")
item("RLHF / preference optimization", "improves helpfulness/safety", "reward hacking; high complexity; evaluation hard")
item("RAG", "injects fresh knowledge; reduces hallucination", "retrieval errors; latency; prompt/grounding complexity")

story.append(Paragraph("Generative models", h3))
item("VAE", "latent structure; likelihood-based", "blurry samples; posterior collapse risk")
item("GAN", "sharp samples", "instability; mode collapse; hard to evaluate")
item("Autoregressive", "good likelihood; controllable", "slow generation; exposure bias")
item("Diffusion", "stable training; SOTA quality", "slow sampling; large compute (mitigations exist)")
item("Normalizing flows", "exact likelihood; invertible", "architectural constraints; memory/compute")

story.append(Paragraph("Reinforcement learning", h3))
item("Q-learning / DQN", "sample efficient (off-policy)", "unstable; overestimation; needs replay/target nets")
item("Policy gradient", "works with continuous actions", "high variance; needs baselines")
item("Actor-critic", "lower variance; efficient", "more moving parts; instability")
item("PPO", "robust default; stable updates", "can be sample hungry; sensitive to reward shaping")
bullets([
    "<b>Core concepts:</b> MDP, reward shaping, discount γ, exploration (ε-greedy/Thompson), off-policy vs on-policy, credit assignment.",
], tiny)

story.append(Paragraph("Time series & forecasting", h3))
item("ARIMA/SARIMA", "interpretable; strong for linear seasonal", "needs stationarity; limited nonlinear patterns")
item("ETS / state-space", "handles trend/seasonality", "model selection; less flexible")
item("Prophet", "easy defaults; holidays", "not best for complex series; can mislead without tuning")
item("RNN/Transformer forecasting", "captures nonlinear + covariates", "needs data; leakage risks; compute")
item("Anomaly detection TS", "detect drift/spikes", "false positives; thresholding; seasonality handling")

story.append(Paragraph("Training tricks & stability", h2))
bullets([
    "<b>Mixed precision:</b> faster; watch underflow/grad scaling.",
    "<b>Gradient clipping:</b> stabilizes RNN/Transformers; can hide LR issues.",
    "<b>Classical tricks:</b> data augmentation, mixup/cutmix, label smoothing, EMA weights.",
    "<b>Monitoring:</b> loss curves, gradient norms, activation stats, overfit gap, calibration drift.",
], tiny)

story.append(Paragraph("Choosing a method (quick heuristics)", h2))
bullets([
    "<b>Tabular:</b> start with GBDT; consider MLP w/ embeddings if lots of categorical + interactions.",
    "<b>Text:</b> pretrained Transformers + fine-tune/PEFT; add RAG if knowledge changes.",
    "<b>Vision:</b> pretrained CNN/ViT; augment; consider distillation for latency.",
    "<b>Graphs:</b> start with features + GBDT; then GNN if relational signal matters.",
    "<b>Small data:</b> linear/trees + strong regularization; transfer learning; uncertainty/GP if feasible.",
], tiny)

doc.build(story)
out_path

