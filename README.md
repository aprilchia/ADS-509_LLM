# ADS 509 — Applied Large Language Models

A research project exploring how transformer-based language models can be used to analyze political discourse across social media platforms. The project pursued two parallel research directions.

**Source classification:** Can a model learn to distinguish between social media platforms based solely on comment style? This experiment treats each platform's community norms and discourse patterns as a fingerprint and asks whether those differences are learnable.

**Discourse labeling (self-labeling experiment):** Do different platforms differ in the *nature* of their discourse — not just *what* people say, but *how* they say it? We built a 5-class labeling scheme (Argumentative, Informational, Opinion, Expressive, Neutral) and used a combination of human annotation and multi-LLM majority voting to label ~70k comments, then fine-tuned transformer models to classify them.

---

## Data Sources

Comments were collected from five platforms around US political discussion:

| Source | Query |
|--------|-------|
| Reddit | r/politics |
| Hacker News | "politics" |
| YouTube | "US Politics" |
| BlueSky | "politics" |
| MetaFilter | politics tag |

**Collection criteria (applied consistently across sources):**
- Date range: Jan 2025 – mid-February 2026
- Posts included only if they had **≥ 10 comments**
- Maximum of **300 comments collected per post**

MetaFilter has a considerably smaller and more niche user base than the other platforms, so the date range was extended further back than our original 6 month filter we placed on the others

---

## Labeling

### Methodology

1. **Manual annotation:** 100 comments were independently labeled by 2 annotators to establish ground truth and refine label definitions.
2. **LLM labeling:** The remaining ~70k comments were labeled via Batch API using three models — Gemini Flash 3, ChatGPT 5.1, and Claude Haiku 4.5. Human-labeled examples were included in the prompt as positive and negative references.
3. **Majority vote:** Only comments where **≥ 2 of 3 models agreed** were retained. The rest were set aside.
4. **Spot check:** 100 LLM-labeled samples were spot-checked for quality.

### Label Definitions

| Label | Description |
|-------|-------------|
| **Argumentative** | Makes specific claims, predictions, or assertions supported by reasoning. Uses evidence, anecdotes, or scenarios to build a case. Key distinction from Opinion: there's an attempt to *persuade or explain why*, not just state a position. |
| **Informational** | Shares facts, data, links, or context relevant to the discussion. Low emotional affect — the comment is trying to *inform*, not convince or react. Includes answering another commenter's question with factual content. Key distinction from Argumentative: presenting information without advocating for a position. |
| **Opinion** | States a value judgment, stance, or take without substantial reasoning. "This is good/bad/wrong/overrated" — the comment *asserts* but doesn't *argue*. Key distinction from Argumentative: no real attempt to persuade or support the claim. Key distinction from Expressive: the comment is making a point, not just reacting. |
| **Expressive** | Emotional reactions, sarcasm, jokes, venting, exclamations. The comment is primarily *expressing feeling* rather than making a point. Includes performative agreement/disagreement ("THIS," "lol exactly," "what a joke"). Key distinction from Opinion: no identifiable stance being taken, just affect. |
| **Neutral** | Clarifying or rhetorical questions, meta-commentary, off-topic remarks. Comments that don't clearly fit the other four categories, including simple factual questions directed at other commenters. |

---

## Preprocessing

Two preprocessing pipelines were used depending on the downstream model type.

**For transformer fine-tuning (BERT, BERTweet):**
- Emojis converted to written text descriptions (e.g., 😀 → "grinning face") to preserve the emotion they convey
- URLs replaced with a `[URL]` token — the presence of a link can be informative about a comment's nature
- HTML tags and entities removed; text lowercased

**For traditional ML baselines (TF-IDF + Logistic Regression):**
- All punctuation removed except `?` and `!`, which carry semantic meaning
- Stopwords removed using the NLTK English set
- Text lemmatized with WordNet — reduces feature space without losing meaning

---

## Data Schema

All sources were normalized to the same column names to enable cross-source EDA and modeling.

**Main posts DataFrame:**

| Column | Description |
|--------|-------------|
| `post_id` | Unique post identifier |
| `post_title` | Title of the post |
| `post_author` | Original author or channel |
| `created_at` | Timestamp of the post |
| `comment_count` | Number of comments on the post |
| `source` | Platform name (e.g., `reddit`, `hacker_news`) |

**Comments DataFrame:**

| Column | Description |
|--------|-------------|
| `post_id` | Foreign key linking to the main post |
| `username` | The commenter |
| `created_at` | Timestamp of the comment |
| `comment_text` | Raw text of the comment |

---

## Repository Structure

```
├── data/                       # Processed datasets
├── eda_logs/                   # EDA output logs and JSON summaries
├── eda_notebooks/              # Full per-source EDA (appendix)
├── labeling/                   # Labeling artifacts
├── scripts/                    # Web scrapers
│   ├── hn_scrape.py            # Hacker News (Algolia API)
│   ├── metafilter_scrape.py    # MetaFilter (HTML parsing)
│   ├── reddit_scrape.py        # Reddit (JSON API)
│   └── youtube_scrape.py       # YouTube (Data API v3)
├── utils/                      # Shared utilities
│   ├── initial_eda.py          # EDA analysis engine
│   ├── metafilter_date_convert.py  # Date normalization for MetaFilter
│   ├── preprocessing_script.py # Text cleaning pipelines
│   ├── pretrain_script.py      # Dataset preparation for fine-tuning
│   ├── probabilites.py         # Softmax utility
│   ├── self_labeling.py        # Multi-LLM majority vote logic
│   └── sleepy.py               # Polite rate-limiting delays
├── main_notebook.ipynb         # EDA summaries + baseline models
├── fine_tuning.ipynb           # Fine-tuning workflows (GPU)
├── evaluation.ipynb            # Model evaluation (GPU)
└── pyproject.toml              # Project dependency management file
```

### Notebook Split

The project is split across three notebooks because fine-tuning and evaluation are GPU-intensive workloads. Separating them made it practical to run EDA and baseline modeling locally on CPU while offloading the transformer training and evaluation to a GPU environment, and to iterate on each stage independently.

---

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires Python ≥ 3.12.

```bash
uv venv
uv sync
```

---

## HuggingFace Artifacts

### Dataset

**[ADS509/full_experiment_labels](https://huggingface.co/datasets/ADS509/full_experiment_labels)**

70,383 preprocessed and labeled social media comments across 5 platforms and 5 comment-type classes. Pre-split into train (49,268), validation (10,557), and test (10,558) sets.

### Models

| Model | Base | Accuracy | F1 Macro | Notes |
|-------|------|----------|----------|-------|
| [experiment_labels_bert_base](https://huggingface.co/ADS509/experiment_labels_bert_base) | bert-base-uncased | 74.44% | 0.7295 | Baseline fine-tuned model |
| [BERTweet-large-self-labeling](https://huggingface.co/ADS509/BERTweet-large-self-labeling) | vinai/bertweet-large | 78.85% | 0.7817 | Best model; +7.2% over bert-base |

BERTweet-large was pre-trained on Twitter data, making it a natural fit for social media comment classification. Both models were fine-tuned for 2 epochs with AdamW (lr=2e-5, warmup 300 steps).

---

## Contributors

- April Chia
- Taylor Kirk
- Tommy Baron
- Celina Velazquez
