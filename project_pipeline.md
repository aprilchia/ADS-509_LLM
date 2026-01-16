# Project Pipeline: Transformer Fine-Tuning for Social Media Text Analysis

**Project Overview:** Web scraping → Data curation → Fine-tuning transformer models → Evaluation  
**Likely Task:** Sentiment analysis (specific direction TBD)  
**Constraint:** CPU-only (no GPU access unless absolutely necessary)

---

## 1. DATA INGESTION / SCRAPING

**Target Sources:** Reddit, Metafilter, Truth Social, Ground News/AllSides, Discord, Threads, Hacker News (select 4-5)

**Scraping Criteria:**
- Political posts from last 6 months
- Minimum 10 comments per post
- Capture post metadata: author, title, text, date, link to original media
- Capture comment metadata: author, text, date, parent comment relationships

**Tools:** BeautifulSoup, platform APIs where available

**Storage:** CSV files (switch to Parquet if size becomes an issue)

**Output:** Raw scraped data per source, stored separately

**Tasks:**
- [ ] Verify scraping success rates per source
- [ ] Document data schema/structure differences across platforms
- [ ] Track ethical scraping constraints (rate limits, robots.txt, API ToS, attribution requirements)

**Open Questions:**
- Which 4-5 sources best balance diversity, access, and scraping feasibility?

---

## 2. DATA VALIDATION & QUALITY CHECK

**Goal:** Ensure data integrity before analysis

**Tasks:**
- [ ] Set up logger to track:
  - Scraping completeness (expected vs actual records)
  - Missing value patterns by source
  - Malformed data (encoding issues, truncated text, etc.)
  - Data quality metrics per source

**Output:** Validated raw datasets, logged quality metrics

---

## 3. INITIAL EDA

**Tools:** NLTK, TextBlob, textstat, pandas, matplotlib/seaborn

**Analyses:**
- Descriptive statistics per source (post count, comment count, time distribution)
- Text characteristics: avg character/word/sentence length, readability scores (grade level)
- Feature presence: @ symbols, URLs (.com), emojis, hashtags
- Initial sentiment patterns (VADER intensity analyzer)
- Temporal patterns: distribution of posts/comments over time (by day of week, hour if available)
- Comment thread structure: average thread depth, replies per comment
- Compare distributions across sources

**Output:** EDA notebook with comparative visualizations, initial data insights

**Open Questions:**
- What patterns suggest which sources might be most informative?
- Any immediate red flags in data quality or coverage?

---

## 4. PREPROCESSING ROUND 1

**Standard Operations:**
- [ ] Deduplicate records (within and across sources)
- [ ] Handle missing values (drop, impute, or flag)
- [ ] Address malformed data
- [ ] Lowercase transformation
- [ ] Remove stop words
- [ ] Remove/handle punctuation
- [ ] Tokenization
- [ ] POS tagging

**Tools:** NLTK, spaCy

**Output:** Preprocessed datasets (v1)

**Open Questions:**
- **Tokenization method evaluation:** How to decide which tokenization approach best fits our data?
- **Stop word selection:** Standard NLTK list? Custom list? Keep domain-specific terms?
- **Punctuation handling:** Keep question marks? Exclamation points? (Could signal sentiment/question vs statement)

---

## 5. EDA ROUND 2

**Goal:** Deeper analysis on preprocessed data to inform modeling decisions

**Analyses:**
- N-gram distributions (unigrams, bigrams, trigrams)
- Feature distributions (POS patterns, entity types if applicable)
- Distribution plots:
  - Token length distributions (post-preprocessing)
  - POS tag distributions by source
  - Label distributions (once labeled) - check for class imbalance
- Vocabulary size and overlap across sources
- Feature correlation matrix: identify redundant or highly correlated engineered features
- Vocabulary overlap: calculate Jaccard similarity between source vocabularies (domain shift analysis)
- Source-specific linguistic patterns: identify n-grams, POS patterns, or features concentrated in specific platforms
- Comparative visualizations (bar plots, distribution plots)

**Output:** EDA notebook (round 2), insights to guide feature engineering

**Open Questions:**
- What patterns emerge that weren't visible in raw data?
- Which features appear most discriminative for our task?

---

## 6. LABELING STRATEGY

**Task Definition:** TBD - likely sentiment analysis (options: political valence, argument structure/quality, other)

**Labeling Approach:**
1. **Manual annotation round:** Each team member independently labels initial sample (n=TBD)
2. **Agreement analysis:** Calculate inter-annotator agreement (Cohen's kappa / Fleiss' kappa)
3. **Resolve disagreements:** Discuss and establish labeling guidelines
4. **Few-shot LLM labeling:** Use manual examples as prompts for LLM to label remaining data
5. **Validation:** Independently spot-check LLM-generated labels (sample size TBD)

**Output:** 
- Labeled dataset with quality metrics
- Annotation guidelines documentation
- Label quality report (agreement scores, LLM validation accuracy)

**Open Questions:**
- **Task selection:** Which specific sentiment analysis direction provides most insight?
- **Label schema:** Binary? Multi-class? Continuous scores?
- **Sample sizes:** How many for initial manual labeling? How many for LLM validation?
- **LLM selection:** Which model for labeling? (GPT-4, Claude, open-source?)

---

## 7. TRAIN / VALIDATION / TEST SPLIT

**Split:** Training / Validation / Holdout Test (percentages TBD)

**Considerations:**
- [ ] Stratification strategy: by label? by source? by time?
- [ ] Hold out entire time period for test set to avoid temporal leakage?

**Output:** Three datasets with documented split rationale

**Open Questions:**
- **Generalization goal:** Are we evaluating within-platform performance or cross-platform generalization?
- **Split percentages:** Standard 70/15/15? Adjust based on dataset size?
- **Stratification factors:** What ensures representative splits given multiple sources and potential class imbalance?

---

## 8. MODEL PREP

**Goal:** Prepare final datasets for modeling

**Tasks:**
- [ ] Merge data from all sources
- [ ] Lemmatization (for TF-IDF baseline ONLY - skip for transformer models)
- [ ] Feature engineering (TBD based on EDA insights)
- [ ] Consider creating multiple dataset versions for comparison

**Note on Lemmatization:**
- **Use for:** TF-IDF baseline (reduces vocabulary sparsity)
- **Skip for:** Transformer fine-tuning (models pre-trained on non-lemmatized text; subword tokenization already handles morphological variations)

**Output:** Model-ready datasets

**Open Questions:**
- **Feature selection:** Which engineered features to include? (e.g., comment length, source platform, time features, readability scores)
- **Preprocessing variations:** Test model performance with vs without stop words, with vs without punctuation?
- **Embeddings:** Any additional embeddings beyond model's native ones? Or rely on fine-tuning?

---

## 9. BASELINE MODEL

**Goal:** Establish performance floor before fine-tuning

**Approaches:**
1. Logistic Regression with TF-IDF vectors (with lemmatization)
2. Zero-shot pre-trained transformer (no fine-tuning)

**Tools:** scikit-learn, Hugging Face transformers

**Output:** Baseline performance metrics for comparison

---

## 10. FINE-TUNING

**Strategy:** Select and compare two models (TBD)

**Model Selection Considerations:**
- Size comparison: smaller vs medium (e.g., DistilBERT vs BERT-base)
- Architecture comparison: different pre-training approaches (e.g., DistilBERT vs DistilRoBERTa)
- Domain specificity: social media-trained vs general (e.g., BERTweet vs general BERT)

**Models (CPU-friendly):**
- DistilBERT (~66M parameters)
- DistilRoBERTa
- MobileBERT
- BERTweet (if doing Twitter data)

**Fine-tuning Approach:**
- **Feature extraction** (default): Freeze entire model, train only classification head
- **Adapter layers** (if needed): Add small trainable modules
- **Partial unfreezing** (if computational budget allows): Start with classifier only, possibly unfreeze last 1-2 transformer blocks

**Tools:** Hugging Face Transformers, potentially spaCy

**Output:** Two fine-tuned models with training logs

**Open Questions:**
- **Two model comparison:** Should we compare two models? If so, what criteria for selecting them? (size, architecture, domain?)
- **Tokenization consideration:** Does BPE vs WordPiece matter for our specific data? (Likely no, but verify during selection)
- **Fine-tuning depth:** Classifier only? How many layers to unfreeze if we go deeper?
- **Hyperparameters:** Learning rate, batch size, epochs, early stopping criteria?
- **Class imbalance:** Do we need weighted loss or oversampling?

---

## 11. EVALUATION & ERROR ANALYSIS

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1 (macro and weighted)
- Confusion matrices
- Per-class performance breakdown
- **Cross-source evaluation:** Train on sources A+B+C, test on source D (test generalization)
- Statistical significance: McNemar's test for model comparisons, bootstrap confidence intervals
- Efficiency metrics: inference time, model size, memory footprint

**Error Analysis:**
- Examine misclassifications by category
- Identify patterns in failures (source-specific? length-dependent? specific label confusions?)
- Analyze failure cases to inform potential iterations

**Tools:** scikit-learn, pandas, matplotlib/seaborn

**Output:** 
- Comprehensive evaluation report comparing all models (baseline + fine-tuned)
- Error analysis documentation
- Recommendations for model selection and potential improvements

**Open Questions:**
- **Comparison framework:** How to weight different metrics for final model selection?
- **Threshold tuning:** For multi-class, do we adjust decision thresholds?

---

## DECISION TRACKING

**Purpose:** Document all major decisions, rationale, and outcomes throughout pipeline

**Format:** Markdown decision log with entries:
- **Decision:** What was decided
- **Date:** When
- **Alternatives considered:** What else was on the table
- **Reasoning/Evidence:** Why this choice (cite EDA results, literature, constraints)
- **Results/Impact:** Filled in after implementation

**Location:** `decisions.md` in project repo

**Automation Opportunities:**
- Experiment tracking: Use MLflow or Weights & Biases for model experiments (hyperparameters, metrics, artifacts)
- Data versioning: Could use DVC if dataset versions proliferate
- Results logging: Automated metric collection and comparison tables

**Manual Documentation:**
- Strategic decisions (task selection, source selection, split strategy)
- Preprocessing choices and their rationale
- Model selection reasoning
- Error analysis insights and action items

---

## EXTRAS (Optional/Future Work)

### Deployment & Monitoring
- Set up periodic re-scraping of sources
- Evaluate model drift over time
- Consider automated retraining pipeline (CI/CD) if drift detected
- Deploy as simple API or Streamlit demo

**Note:** Full CI/CD pipeline may be beyond class project scope. Focus on demonstrating concept of monitoring and drift detection.
