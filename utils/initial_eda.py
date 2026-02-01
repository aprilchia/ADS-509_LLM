
# Commented out comment_count, need to create method to calculate that for the other sources first
# Have day of week created in here


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import textstat  # type: ignore
import re
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime as dt
from IPython.display import display, HTML
from textblob import TextBlob  # type: ignore

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

try:
    from tqdm import tqdm
    tqdm.pandas()
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import emoji as emoji_lib  # type: ignore
    _HAS_EMOJI_LIB = True
except ImportError:
    _HAS_EMOJI_LIB = False

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_MAIN_COLS = {"post_title", "post_id", "post_author", "created_at"} # "comment_count"}
REQUIRED_COMMENT_COLS = {"post_id", "username", "comment_text", "created_at"}

POS_GROUPS = {
    "adj":          {"JJ", "JJR", "JJS"},
    "adv":          {"RB", "RBR", "RBS"},
    "pronoun":      {"PRP", "PRP$"},
    "modal":        {"MD"},
    "interjection": {"UH"},
}

POS_GROUP_LABELS = {
    "adj":          "Adjectives (JJ*)",
    "adv":          "Adverbs (RB*)",
    "pronoun":      "Pronouns (PRP*)",
    "modal":        "Modals (MD)",
    "interjection": "Interjections (UH)",
}


# ---------------------------------------------------------------------------
# EDAReport dataclass
# ---------------------------------------------------------------------------
@dataclass
class EDAReport:
    source: str
    main_enriched: pd.DataFrame
    comments_enriched: pd.DataFrame
    summary: Dict[str, Any] = field(default_factory=dict)
    descriptive_stats: Dict[str, Any] = field(default_factory=dict)
    text_characteristics: Dict[str, Any] = field(default_factory=dict)
    readability_scores: Dict[str, Any] = field(default_factory=dict)
    feature_presence: Dict[str, Any] = field(default_factory=dict)
    sentiment_summary: Dict[str, Any] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    thread_structure: Dict[str, Any] = field(default_factory=dict)
    pos_distribution: Dict[str, Any] = field(default_factory=dict)
    figures: List[Tuple[str, plt.Figure]] = field(default_factory=list) # type: ignore

    def to_log_dict(self) -> Dict[str, Any]:
        """Merge all section dicts into one flat dict for JSON logging.
        Excludes list values (e.g. feature indexes) to keep JSON compact."""
        merged: Dict[str, Any] = {"source": self.source}
        for d in [
            self.descriptive_stats,
            self.text_characteristics,
            self.readability_scores,
            self.feature_presence,
            self.sentiment_summary,
            self.temporal_patterns,
            self.thread_structure,
            self.pos_distribution,
        ]:
            for k, v in d.items():
                if not isinstance(v, list):
                    merged[k] = v
        return merged


# ---------------------------------------------------------------------------
# Helper: progress-aware apply
# ---------------------------------------------------------------------------
def _apply(series: pd.Series, func, desc: str = "Processing"):  # type: ignore[type-arg]
    if _HAS_TQDM:
        tqdm.pandas(desc=desc)  # type: ignore[possibly-undefined]
        return series.progress_apply(func)  # type: ignore[attr-defined]
    return series.apply(func)


# ---------------------------------------------------------------------------
# Helper: column validation
# ---------------------------------------------------------------------------
def _validate_columns(main: pd.DataFrame, comments: pd.DataFrame):
    missing_main = REQUIRED_MAIN_COLS - set(main.columns)
    missing_comments = REQUIRED_COMMENT_COLS - set(comments.columns)
    if missing_main or missing_comments:
        msg = ""
        if missing_main:
            msg += f"Main df missing required columns: {missing_main}\n"
        if missing_comments:
            msg += f"Comments df missing required columns: {missing_comments}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Helper: display utilities
# ---------------------------------------------------------------------------
def _section_header(title: str):
    display(HTML(
        f'<h2 style="color:#2c3e50; border-bottom:2px solid #3498db; '
        f'padding-bottom:6px; margin-top:28px;">{title}</h2>'
    ))


def _sub_header(title: str):
    display(HTML(
        f'<h3 style="color:#34495e; margin-top:16px;">{title}</h3>'
    ))


def _display_df(df: pd.DataFrame, caption: str = ""):
    if caption:
        display(HTML(f'<p style="font-weight:600; color:#34495e;">{caption}</p>'))
    display(df)


def _display_metric(label: str, value: Any) -> None:
    display(HTML(
        f'<p style="margin:2px 0;"><strong>{label}:</strong> {value}</p>'
    ))


def _display_figure(fig: plt.Figure, report: EDAReport, label: str, # type: ignore
                    save_dir: Optional[str] = None, show: bool = True):
    report.figures.append((label, fig))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{report.source}_{label}.png"),
            dpi=150, bbox_inches="tight",
        )
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: batched text statistics
# ---------------------------------------------------------------------------
def _compute_text_stats(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "char_length": 0, "word_count": 0, "sentence_count": 0,
            "avg_word_length": np.nan, "syllable_count": 0,
            "lexicon_count": 0, "polysyllable_count": 0,
            "monosyllable_count": 0, "flesch_reading_ease": np.nan,
            "dale_chall": np.nan, "grade_level": np.nan, "gunning_fog": np.nan,
        }
    text = str(text)
    words = text.split()
    char_len = len(text)
    word_cnt = len(words)
    return {
        "char_length": char_len,
        "word_count": word_cnt,
        "sentence_count": textstat.sentence_count(text),
        "avg_word_length": char_len / word_cnt if word_cnt > 0 else 0,
        "syllable_count": textstat.syllable_count(text),
        "lexicon_count": textstat.lexicon_count(text),
        "polysyllable_count": textstat.polysyllabcount(text),
        "monosyllable_count": textstat.monosyllabcount(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "dale_chall": textstat.dale_chall_readability_score_v2(text),
        "grade_level": textstat.text_standard(text, float_output=True),
        "gunning_fog": textstat.gunning_fog(text),
    }


def _ensure_text_stats(report: EDAReport):
    """Compute text stats columns if they haven't been computed yet."""
    if "char_length" in report.comments_enriched.columns:
        return
    text = report.comments_enriched["comment_text"].fillna("")
    stats = _apply(text, _compute_text_stats, "Computing text statistics")
    stats_df = pd.DataFrame(stats.tolist())
    for col in stats_df.columns:
        report.comments_enriched[col] = stats_df[col]


# ---------------------------------------------------------------------------
# 1. Descriptive Statistics
# ---------------------------------------------------------------------------
def analyze_descriptive_stats(report: EDAReport, show_plots: bool = True,
                              save_dir: Optional[str] = None):
    _section_header(f"1. Descriptive Statistics  ({report.source})")

    main = report.main_enriched
    comments = report.comments_enriched

    # Basic counts
    post_count = len(main)
    comment_count = len(comments)
    unique_posters = main["post_author"].nunique()
    unique_commenters = comments["username"].nunique()

    # Date range from main
    try:
        dates = pd.to_datetime(main["created_at"])
        date_start = dates.min().strftime("%Y-%m-%d")
        date_end = dates.max().strftime("%Y-%m-%d")
    except Exception:
        date_start = main["created_at"].min()
        date_end = main["created_at"].max()

    summary_data = {
        "Posts": post_count,
        "Comments": comment_count,
        "Unique Post Authors": unique_posters,
        "Unique Commenters": unique_commenters,
        "Date Range": f"{date_start}  to  {date_end}",
    }
    summary_df = pd.DataFrame.from_dict(summary_data, orient="index", columns=["Value"])
    _display_df(summary_df, "Overview")

    # Comments per post
    cpt = comments.groupby("post_id").size()
    cpt_stats = cpt.describe()
    _display_df(
        pd.DataFrame(cpt_stats).rename(columns={0: "Comments Per Post"}),
        "Comments Per Post Distribution",
    )

    # ---- Missing values heatmaps ----
    _sub_header("Missing Values")
    has_missing = main.isnull().any().any() or comments.isnull().any().any()
    if has_missing:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.heatmap(main.isnull(), cbar=False, yticklabels=False, ax=axes[0],
                    cmap="YlOrRd")
        axes[0].set_title("Main DataFrame")
        sns.heatmap(comments.drop(columns=["favorites_int"], errors="ignore").isnull(),
                    cbar=False, yticklabels=False, ax=axes[1], cmap="YlOrRd")
        axes[1].set_title("Comments DataFrame")
        fig.suptitle("Missing Values Heatmap", fontsize=14, y=1.02)
        fig.tight_layout()
        _display_figure(fig, report, "missing_values", save_dir, show_plots)
    else:
        display(HTML('<p style="color:green;">No missing values in either DataFrame.</p>'))

    # ---- Comments per post histogram ----
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(cpt, bins=40, kde=True, edgecolor="black", ax=ax) # type: ignore
    ax.set_title("Comments Per Post Distribution")
    ax.set_xlabel("Number of Comments")
    ax.set_ylabel("Count")
    fig.tight_layout()
    _display_figure(fig, report, "comments_per_post", save_dir, show_plots)

    # Store results
    report.descriptive_stats = {
        "post_count": post_count,
        "comment_count": comment_count,
        "unique_post_authors": unique_posters,
        "unique_commenters": unique_commenters,
        "date_range_start": str(date_start),
        "date_range_end": str(date_end),
        "comments_per_post_mean": float(cpt.mean()),
        "comments_per_post_median": float(cpt.median()),
        #"avg_favorites": float(fav.mean()),
        #"median_favorites": float(fav.median()),
    }


# ---------------------------------------------------------------------------
# 2. Text Characteristics
# ---------------------------------------------------------------------------
def analyze_text_characteristics(report: EDAReport, show_plots: bool = True,
                                 save_dir: Optional[str] = None):
    _section_header(f"2. Text Characteristics  ({report.source})")

    _ensure_text_stats(report)
    comments = report.comments_enriched

    # Summary table
    text_cols = [
        "char_length", "word_count", "sentence_count", "avg_word_length",
        "syllable_count", "lexicon_count", "polysyllable_count", "monosyllable_count",
    ]
    summary = comments[text_cols].describe().T[["mean", "50%", "std", "min", "max"]]
    summary = summary.rename(columns={"50%": "median"}).round(2)
    _display_df(summary, "Text Metrics Summary")

    # 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    labels = [
        "Character Length", "Word Count", "Sentence Count", "Avg Word Length",
        "Syllable Count", "Lexicon Count", "Polysyllable Count", "Monosyllable Count",
    ]
    for idx, (col, label) in enumerate(zip(text_cols, labels)):
        ax = axes.flat[idx]
        data = comments[col].dropna()
        sns.histplot(data, bins=50, kde=True, edgecolor="black", ax=ax) # type: ignore
        ax.set_xlabel(label)
        ax.set_ylabel("")
    fig.suptitle("Text Characteristics Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    _display_figure(fig, report, "text_characteristics", save_dir, show_plots)

    report.text_characteristics = {
        "avg_char_length": float(comments["char_length"].mean()),
        "avg_word_count": float(comments["word_count"].mean()),
        "avg_sentence_count": float(comments["sentence_count"].mean()),
        "median_char_length": float(comments["char_length"].median()),
        "median_word_count": float(comments["word_count"].median()),
        "avg_avg_word_length": float(comments["avg_word_length"].mean()),
    }


# ---------------------------------------------------------------------------
# 3. Readability Scores
# ---------------------------------------------------------------------------
def analyze_readability(report: EDAReport, show_plots: bool = True,
                        save_dir: Optional[str] = None):
    _section_header(f"3. Readability Scores  ({report.source})")

    _ensure_text_stats(report)
    comments = report.comments_enriched

    read_cols = ["flesch_reading_ease", "dale_chall", "grade_level", "gunning_fog"]
    labels = ["Flesch Reading Ease", "Dale-Chall", "Grade Level", "Gunning Fog"]

    # Summary
    summary = comments[read_cols].describe().T[["mean", "50%", "std", "min", "max"]]
    summary = summary.rename(columns={"50%": "median"}).round(2)
    _display_df(summary, "Readability Scores Summary")

    # Interpretation table
    display(HTML("""
    <table style="border-collapse:collapse; margin:8px 0; border:1px solid #bdc3c7;">
      <tr style="background:#34495e;">
          <th style="padding:6px 12px; text-align:left; color:#ecf0f1;">Metric</th>
          <th style="padding:6px 12px; text-align:left; color:#ecf0f1;">Scale Interpretation</th></tr>
      <tr style="background:#fdfefe;">
          <td style="padding:4px 12px; color:#2c3e50;">Flesch Reading Ease</td>
          <td style="padding:4px 12px; color:#2c3e50;">0-30 Very Difficult | 30-50 Difficult | 50-60 Fairly Difficult | 60-70 Standard | 70-80 Fairly Easy | 80-90 Easy | 90+ Very Easy</td></tr>
      <tr style="background:#eaf2f8;">
          <td style="padding:4px 12px; color:#2c3e50;">Dale-Chall</td>
          <td style="padding:4px 12px; color:#2c3e50;">4.9 or lower = Grade 4 | 5.0-5.9 = Grades 5-6 | 6.0-6.9 = Grades 7-8 | 7.0-7.9 = Grades 9-10 | 8.0-8.9 = Grades 11-12 | 9.0+ = College</td></tr>
      <tr style="background:#fdfefe;">
          <td style="padding:4px 12px; color:#2c3e50;">Grade Level</td>
          <td style="padding:4px 12px; color:#2c3e50;">Consensus US grade level (average of multiple formulas)</td></tr>
      <tr style="background:#eaf2f8;">
          <td style="padding:4px 12px; color:#2c3e50;">Gunning Fog</td>
          <td style="padding:4px 12px; color:#2c3e50;">6 = Easy | 8 = Average | 12 = High school senior | 17+ = College graduate</td></tr>
    </table>
    """))

    # 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (col, label) in enumerate(zip(read_cols, labels)):
        ax = axes.flat[idx]
        data = comments[col].dropna()
        sns.histplot(data, bins=50, kde=True, edgecolor="black", ax=ax) # type: ignore
        ax.set_xlabel(label)
        ax.set_ylabel("")
    fig.suptitle("Readability Score Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    _display_figure(fig, report, "readability_scores", save_dir, show_plots)

    report.readability_scores = {
        "avg_flesch_reading_ease": float(comments["flesch_reading_ease"].mean()),
        "avg_dale_chall": float(comments["dale_chall"].mean()),
        "avg_grade_level": float(comments["grade_level"].mean()),
        "avg_gunning_fog": float(comments["gunning_fog"].mean()),
    }


# ---------------------------------------------------------------------------
# 4. Feature Presence
# ---------------------------------------------------------------------------
def _compute_feature_presence(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "has_mention": False, "mention_count": 0,
            "has_url": False, "url_count": 0,
            "has_hashtag": False, "hashtag_count": 0,
            "has_emoji": False, "emoji_count": 0,
        }
    text = str(text)
    mentions = re.findall(r"@\w+", text)
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    hashtags = re.findall(r"#\w+", text)

    if _HAS_EMOJI_LIB:
        em_count = emoji_lib.emoji_count(text)  # type: ignore[possibly-undefined]
    else:
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
            "\U00002702-\U000027B0]+"
        )
        em_count = len(emoji_pattern.findall(text))

    return {
        "has_mention": len(mentions) > 0,
        "mention_count": len(mentions),
        "has_url": len(urls) > 0,
        "url_count": len(urls),
        "has_hashtag": len(hashtags) > 0,
        "hashtag_count": len(hashtags),
        "has_emoji": em_count > 0,
        "emoji_count": em_count,
    }


def analyze_feature_presence(report: EDAReport, show_plots: bool = True,
                             save_dir: Optional[str] = None):
    _section_header(f"4. Feature Presence  ({report.source})")

    comments = report.comments_enriched

    if "has_mention" not in comments.columns:
        feats = _apply(
            comments["comment_text"].fillna(""),
            _compute_feature_presence,
            "Detecting features",
        )
        feat_df = pd.DataFrame(feats.tolist())
        for col in feat_df.columns:
            comments[col] = feat_df[col]

    n = len(comments)
    feature_names = ["Mentions (@)", "URLs", "Hashtags (#)", "Emojis"]
    has_cols = ["has_mention", "has_url", "has_hashtag", "has_emoji"]
    count_cols = ["mention_count", "url_count", "hashtag_count", "emoji_count"]

    pcts = [comments[c].sum() / n * 100 for c in has_cols]

    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "% Comments With": [f"{p:.1f}%" for p in pcts],
        "Avg Count (when present)": [
            f"{comments.loc[comments[hc], cc].mean():.2f}"
            if comments[hc].sum() > 0 else "N/A"
            for hc, cc in zip(has_cols, count_cols)
        ],
    }).set_index("Feature")
    _display_df(summary_df, "Feature Presence Summary")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("muted", len(feature_names))
    ax.barh(feature_names, pcts, color=colors, edgecolor="black")
    ax.set_xlabel("% of Comments")
    ax.set_title("Feature Presence in Comments")
    for i, v in enumerate(pcts):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    _display_figure(fig, report, "feature_presence", save_dir, show_plots)

    report.feature_presence = {
        "pct_with_mentions": pcts[0],
        "pct_with_urls": pcts[1],
        "pct_with_hashtags": pcts[2],
        "pct_with_emojis": pcts[3],
        "mention_indexes": comments.index[comments["has_mention"]].tolist(),
        "url_indexes": comments.index[comments["has_url"]].tolist(),
        "hashtag_indexes": comments.index[comments["has_hashtag"]].tolist(),
        "emoji_indexes": comments.index[comments["has_emoji"]].tolist(),
    }


# ---------------------------------------------------------------------------
# 5. VADER Sentiment
# ---------------------------------------------------------------------------
def analyze_sentiment(report: EDAReport, show_plots: bool = True,
                      save_dir: Optional[str] = None):
    _section_header(f"5. VADER Sentiment Analysis  ({report.source})")

    comments = report.comments_enriched

    if "vader_compound" not in comments.columns:
        sia = SentimentIntensityAnalyzer()

        def _vader_scores(text):
            if not text or not str(text).strip():
                return {"compound": np.nan, "pos": np.nan, "neg": np.nan, "neu": np.nan}
            return sia.polarity_scores(str(text))

        scores = _apply(
            comments["comment_text"].fillna(""),
            _vader_scores,
            "Computing VADER sentiment",
        )
        scores_df = pd.DataFrame(scores.tolist())
        comments["vader_compound"] = scores_df["compound"]
        comments["vader_pos"] = scores_df["pos"]
        comments["vader_neg"] = scores_df["neg"]
        comments["vader_neu"] = scores_df["neu"]
        comments["vader_label"] = comments["vader_compound"].apply(
            lambda x: "positive" if x >= 0.05
            else ("negative" if x <= -0.05 else "neutral")
        )

    compound = comments["vader_compound"].dropna()

    # Summary stats
    label_counts = comments["vader_label"].value_counts()
    n_valid = label_counts.sum()
    _display_df(
        pd.DataFrame({
            "Count": label_counts,
            "Percentage": (label_counts / n_valid * 100).round(1).astype(str) + "%",
        }),
        "Sentiment Distribution",
    )
    _display_metric("Mean Compound Score", f"{compound.mean():.4f}")
    _display_metric("Median Compound Score", f"{compound.median():.4f}")

    # Compound histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(compound, bins=50, kde=True, edgecolor="black", ax=axes[0]) # type: ignore
    axes[0].axvline(0.05, color="green", linestyle="--", alpha=0.7, label="Positive threshold")
    axes[0].axvline(-0.05, color="red", linestyle="--", alpha=0.7, label="Negative threshold")
    axes[0].set_title("VADER Compound Score Distribution")
    axes[0].set_xlabel("Compound Score")
    axes[0].legend()

    # Label distribution bar chart
    colors_map = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
    bar_colors = [colors_map.get(lbl, "#95a5a6") for lbl in label_counts.index]
    axes[1].bar(label_counts.index, label_counts.values, color=bar_colors, edgecolor="black")
    axes[1].set_title("Sentiment Label Distribution")
    axes[1].set_ylabel("Count")
    for i, (label, count) in enumerate(label_counts.items()):
        axes[1].text(i, count + n_valid * 0.01, str(count), ha="center")
    fig.tight_layout()
    _display_figure(fig, report, "vader_sentiment", save_dir, show_plots)

    # Top positive and negative comments
    valid_comments = comments.dropna(subset=["vader_compound"])
    if len(valid_comments) > 0:
        _sub_header("Top 5 Most Positive Comments")
        top_pos = valid_comments.nlargest(5, "vader_compound")[
            ["comment_text", "vader_compound"]
        ].copy()
        top_pos["comment_text"] = top_pos["comment_text"].str[:200] + "..."
        _display_df(top_pos.reset_index(drop=True))

        _sub_header("Top 5 Most Negative Comments")
        top_neg = valid_comments.nsmallest(5, "vader_compound")[
            ["comment_text", "vader_compound"]
        ].copy()
        top_neg["comment_text"] = top_neg["comment_text"].str[:200] + "..."
        _display_df(top_neg.reset_index(drop=True))

    report.sentiment_summary = {
        "avg_compound": float(compound.mean()),
        "median_compound": float(compound.median()),
        "std_compound": float(compound.std()),
        "pct_positive": float((label_counts.get("positive", 0) / n_valid) * 100),
        "pct_negative": float((label_counts.get("negative", 0) / n_valid) * 100),
        "pct_neutral": float((label_counts.get("neutral", 0) / n_valid) * 100),
    }


# ---------------------------------------------------------------------------
# 6. Temporal Patterns
# ---------------------------------------------------------------------------
def analyze_temporal_patterns(report: EDAReport, show_plots: bool = True,
                              save_dir: Optional[str] = None):
    _section_header(f"6. Temporal Patterns  ({report.source})")

    comments = report.comments_enriched

    # Use created_at column directly
    if "created_at" not in comments.columns:
        display(HTML('<p style="color:orange;">No created_at column found. '
                     'Skipping temporal analysis.</p>'))
        return

    # Ensure created_at is datetime type
    comments["created_at"] = pd.to_datetime(comments["created_at"])

    # Add day_of_week column to comments_enriched
    comments["day_of_week"] = comments["created_at"].dt.day_name() # type: ignore

    valid = comments.dropna(subset=["created_at"])
    if len(valid) == 0:
        display(HTML('<p style="color:orange;">No valid datetimes found. '
                     'Skipping temporal visualizations.</p>'))
        return

    valid_dt = valid["created_at"]

    # Day of week
    dow = valid_dt.dt.day_name() # type: ignore
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    dow_counts = dow.value_counts().reindex(dow_order, fill_value=0)

    # Hour of day
    hour_counts = valid_dt.dt.hour.value_counts().sort_index() # type: ignore

    # Volume over time (by date)
    date_counts = valid_dt.dt.date.value_counts().sort_index() # type: ignore

    # Most active
    most_active_day = dow_counts.idxmax() if len(dow_counts) > 0 else "N/A"
    most_active_hour = int(hour_counts.idxmax()) if len(hour_counts) > 0 else None

    _display_metric("Most Active Day of Week", most_active_day)
    if most_active_hour is not None:
        _display_metric("Most Active Hour", f"{most_active_hour}:00")
    _display_metric("Comments With Valid Dates",
                    f"{len(valid)} / {len(comments)} ({len(valid)/len(comments)*100:.1f}%)")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Day of week
    sns.barplot(x=dow_counts.index, y=dow_counts.values, hue=dow_counts.index,
                palette="muted", edgecolor="black", legend=False, ax=axes[0])
    axes[0].set_title("Comments by Day of Week")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Hour of day
    sns.barplot(x=hour_counts.index, y=hour_counts.values, hue=hour_counts.index,
                palette="muted", edgecolor="black", legend=False, ax=axes[1])
    axes[1].set_title("Comments by Hour of Day")
    axes[1].set_xlabel("Hour (24h)")
    axes[1].set_ylabel("Count")

    # Volume over time
    axes[2].plot(date_counts.index, date_counts.values, marker="o", markersize=3)
    axes[2].set_title("Comment Volume Over Time")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Comments")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _display_figure(fig, report, "temporal_patterns", save_dir, show_plots)

    report.temporal_patterns = {
        "most_active_day": most_active_day,
        "most_active_hour": most_active_hour,
        "comments_per_day_mean": float(date_counts.mean()) if len(date_counts) > 0 else 0,
        "date_range_days": (valid_dt.max() - valid_dt.min()).days,
        "pct_valid_dates": float(len(valid) / len(comments) * 100),
    }


# ---------------------------------------------------------------------------
# 7. Thread Structure
# ---------------------------------------------------------------------------
def analyze_thread_structure(report: EDAReport, show_plots: bool = True,
                             save_dir: Optional[str] = None):
    _section_header(f"7. Thread Structure  ({report.source})")

    main = report.main_enriched
    comments = report.comments_enriched

    # Comments per thread
    cpt = comments.groupby("post_id").size().reset_index(name="n_comments")

    # Add comment_count to main_enriched
    report.main_enriched = report.main_enriched.merge(
        cpt.rename(columns={"n_comments": "comment_count"}),
        on="post_id",
        how="left",
    )
    report.main_enriched["comment_count"] = (
        report.main_enriched["comment_count"].fillna(0).astype(int)
    )

    # Unique commenters per thread
    uct = (
        comments.groupby("post_id")["username"]
        .nunique()
        .reset_index(name="unique_commenters")
    )

    thread_stats = cpt.merge(uct, on="post_id")
    thread_stats["comment_commenter_ratio"] = (
        thread_stats["n_comments"] / thread_stats["unique_commenters"]
    )

    # Summary
    _display_df(
        thread_stats[["n_comments", "unique_commenters", "comment_commenter_ratio"]]
        .describe().round(2),
        "Thread Structure Summary",
    )

    # Top 10 threads
    top_threads = thread_stats.nlargest(10, "n_comments").merge(
        main[["post_id", "post_title"]], on="post_id", how="left"
    )
    top_threads["title_short"] = top_threads["post_title"].fillna("(No Title)").str[:60]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Comments per thread histogram
    sns.histplot(thread_stats["n_comments"], bins=40, kde=True, edgecolor="black", # type: ignore
                 ax=axes[0])
    axes[0].set_title("Comments Per Thread Distribution")
    axes[0].set_xlabel("Number of Comments")

    # Top 10 threads
    axes[1].barh(
        top_threads["title_short"],
        top_threads["n_comments"],
        color=sns.color_palette("muted", len(top_threads)),
        edgecolor="black",
    )
    axes[1].set_title("Top 10 Most Discussed Threads")
    axes[1].set_xlabel("Number of Comments")
    axes[1].invert_yaxis()

    fig.tight_layout()
    _display_figure(fig, report, "thread_structure", save_dir, show_plots)

    report.thread_structure = {
        "avg_comments_per_thread": float(thread_stats["n_comments"].mean()),
        "median_comments_per_thread": float(thread_stats["n_comments"].median()),
        "max_comments_per_thread": int(thread_stats["n_comments"].max()),
        "avg_unique_commenters": float(thread_stats["unique_commenters"].mean()),
        "avg_comment_commenter_ratio": float(
            thread_stats["comment_commenter_ratio"].mean()
        ),
    }


# ---------------------------------------------------------------------------
# 8. POS Tag Distributions
# ---------------------------------------------------------------------------
def _compute_pos_ratios(text: str) -> Dict[str, Any]:
    nan_result = {f"{group}_ratio": np.nan for group in POS_GROUPS}
    if not text or not str(text).strip():
        return nan_result
    try:
        tags = TextBlob(str(text)).tags
    except Exception:
        return nan_result
    total = len(tags)
    if total == 0:
        return nan_result
    counts = {group: 0 for group in POS_GROUPS}
    for _, tag in tags:
        for group, tag_set in POS_GROUPS.items():
            if tag in tag_set:
                counts[group] += 1
    return {f"{group}_ratio": counts[group] / total for group in POS_GROUPS}


def analyze_pos_distribution(report: EDAReport, show_plots: bool = True,
                             save_dir: Optional[str] = None):
    _section_header(f"8. POS Tag Distributions  ({report.source})")
    display(HTML(
        '<p style="color:#7f8c8d; font-style:italic;">'
        'Analyzing 5 sentiment-relevant POS groups: Adjectives, Adverbs, '
        'Pronouns, Modals, Interjections (as proportion of total tokens).</p>'
    ))

    comments = report.comments_enriched
    ratio_cols = [f"{g}_ratio" for g in POS_GROUPS]

    if ratio_cols[0] not in comments.columns:
        pos_results = _apply(
            comments["comment_text"].fillna(""),
            _compute_pos_ratios,
            "Computing POS tag ratios",
        )
        pos_df = pd.DataFrame(pos_results.tolist())
        for col in pos_df.columns:
            comments[col] = pos_df[col]

    # Summary table
    summary = comments[ratio_cols].describe().T[["mean", "50%", "std"]].round(4)
    summary = summary.rename(columns={"50%": "median"})
    summary.index = [POS_GROUP_LABELS[g] for g in POS_GROUPS]
    _display_df(summary, "POS Group Ratios (proportion of total tokens)")

    # Mean ratios bar chart + boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of means
    means = comments[ratio_cols].mean()
    labels = [POS_GROUP_LABELS[g] for g in POS_GROUPS]
    colors = sns.color_palette("muted", len(labels))
    axes[0].barh(labels, means.values, color=colors, edgecolor="black")
    axes[0].set_xlabel("Mean Ratio (proportion of tokens)")
    axes[0].set_title("Mean POS Group Ratios Across All Comments")
    for i, v in enumerate(means.values):
        axes[0].text(v + 0.001, i, f"{v:.3f}", va="center")

    # Boxplots
    box_data = comments[ratio_cols].dropna()
    box_data.columns = labels
    sns.boxplot(data=box_data, orient="h", ax=axes[1], palette="muted")
    axes[1].set_xlabel("Ratio")
    axes[1].set_title("POS Group Ratio Distributions")

    fig.tight_layout()
    _display_figure(fig, report, "pos_distribution", save_dir, show_plots)

    report.pos_distribution = {
        f"mean_{g}_ratio": float(comments[f"{g}_ratio"].mean())
        for g in POS_GROUPS
    }
    report.pos_distribution.update({
        f"median_{g}_ratio": float(comments[f"{g}_ratio"].median())
        for g in POS_GROUPS
    })


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def initial_eda(
    main: pd.DataFrame,
    comments: pd.DataFrame,
    source: str,
    sections: Optional[List[str]] = None,
    show_plots: bool = True,
    save_plots: Optional[str] = None,
) -> EDAReport:
    """
    Run initial EDA on a source's posts and comments DataFrames.

    Parameters
    ----------
    main : pd.DataFrame
        Posts/threads DataFrame with columns matching REQUIRED_MAIN_COLS.
    comments : pd.DataFrame
        Comments DataFrame with columns matching REQUIRED_COMMENT_COLS.
    source : str
        Name of the data source (e.g., "Metafilter", "HackerNews").
    sections : list, optional
        Subset of sections to run. Options: "descriptive", "text",
        "readability", "features", "sentiment", "temporal", "thread", "pos".
        Default: run all.
    show_plots : bool
        If True, display plots inline. If False, only store in report.
    save_plots : str, optional
        Directory to save figure PNGs.

    Returns
    -------
    EDAReport
        Dataclass with enriched DataFrames, summary dicts, and figures.
    """
    _validate_columns(main, comments)

    report = EDAReport(
        source=source,
        main_enriched=main.copy(),
        comments_enriched=comments.copy(),
    )

    # Fill NaN text to prevent errors
    report.comments_enriched["comment_text"] = (
        report.comments_enriched["comment_text"].fillna("")
    )
    report.main_enriched = report.main_enriched.fillna("")

    all_sections = {
        "descriptive": analyze_descriptive_stats,
        "text":        analyze_text_characteristics,
        "readability": analyze_readability,
        "features":    analyze_feature_presence,
        "sentiment":   analyze_sentiment,
        "temporal":    analyze_temporal_patterns,
        "thread":      analyze_thread_structure,
        "pos":         analyze_pos_distribution,
    }

    run_sections = sections or list(all_sections.keys())

    # Title
    display(HTML(
        f'<h1 style="color:#2c3e50; border-bottom:3px solid #2c3e50; '
        f'padding-bottom:8px;">Initial EDA Report: {source}</h1>'
    ))

    for name in run_sections:
        if name in all_sections:
            all_sections[name](report, show_plots, save_plots)
        else:
            display(HTML(
                f'<p style="color:orange;">Unknown section: "{name}". Skipping.</p>'
            ))

    report.summary = report.to_log_dict()
    return report


# ---------------------------------------------------------------------------
# Logging & Persistence
# ---------------------------------------------------------------------------
def save_eda_report(report: EDAReport, output_dir: str = "eda_logs"):
    """Persist EDA report artifacts to disk in a source-specific subfolder."""
    source_dir = os.path.join(output_dir, report.source)
    os.makedirs(source_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{report.source}_{timestamp}"

    # 1. JSON summary (exclude list values like feature indexes)
    log_dict = report.to_log_dict()
    log_dict["timestamp"] = timestamp
    log_dict["main_shape"] = list(report.main_enriched.shape)
    log_dict["comments_shape"] = list(report.comments_enriched.shape)

    json_path = os.path.join(source_dir, f"{prefix}_summary.json")
    with open(json_path, "w") as f:
        json.dump(log_dict, f, indent=2, default=str)

    # 2. Enriched comments CSV
    csv_path = os.path.join(source_dir, f"{prefix}_comments_enriched.csv")
    report.comments_enriched.to_csv(csv_path, index=False)

    # 3. Figures
    for label, fig in report.figures:
        fig_path = os.path.join(source_dir, f"{prefix}_{label}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    display(HTML(
        f'<p style="color:green;">Report saved to <code>{source_dir}/</code> '
        f'with prefix <code>{prefix}</code></p>'
    ))


def compare_sources(log_dir: str = "eda_logs") -> pd.DataFrame:
    """Load all saved EDA summaries and return a comparison DataFrame."""
    import glob as glob_mod
    files = glob_mod.glob(os.path.join(log_dir, "**", "*_summary.json"),
                          recursive=True)
    if not files:
        print(f"No summary files found in {log_dir}/")
        return pd.DataFrame()
    records = []
    for f in files:
        with open(f) as fh:
            records.append(json.load(fh))
    return pd.DataFrame(records).set_index("source")
