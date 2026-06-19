"""
Caption text analysis — finds words, phrases, and patterns that
correlate with higher engagement in the creator's own post history.
"""
import re
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from config import TOP_POSTS_COUNT


STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","it","this","that","are","was","be","as","by","from","have","has",
    "i","my","we","our","you","your","me","he","she","they","their","its",
    "not","so","if","do","did","can","will","just","get","got","all","been",
    "up","out","no","one","more","than","then","into","about","also","when",
    "what","how","who","which","there","here","her","him","his","had","were",
}


def _tokenise(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # strip URLs
    text = re.sub(r"#\w+", "", text)             # strip hashtags
    text = re.sub(r"@\w+", "", text)             # strip mentions
    tokens = re.findall(r"[a-z]{3,}", text)      # words ≥ 3 chars
    return [t for t in tokens if t not in STOP_WORDS]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def caption_word_stats(posts: pd.DataFrame) -> pd.DataFrame:
    """Return per-caption stats: word count, has_question, has_cta, has_emoji."""
    df = posts.copy()
    df["caption"] = df["caption"].fillna("")
    df["word_count"] = df["caption"].apply(lambda c: len(str(c).split()))
    df["char_count"] = df["caption"].apply(len)
    df["has_question"] = df["caption"].str.contains(r"\?", na=False)
    df["has_cta"] = df["caption"].str.contains(
        r"\b(comment|save|share|link in bio|dm me|follow|tag|click)\b",
        case=False, na=False
    )
    df["has_emoji"] = df["caption"].str.contains(
        r"[\U0001F300-\U0001FFFF]", na=False
    )
    df["hashtag_count"] = df["caption"].str.count(r"#\w+")
    return df


def top_words_by_engagement(posts: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Words that appear most frequently in above-average-engagement posts."""
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    threshold = df["engagement"].median()
    high = df[df["engagement"] >= threshold]["caption"].fillna("")
    low = df[df["engagement"] < threshold]["caption"].fillna("")

    high_words = Counter()
    low_words = Counter()
    for text in high:
        high_words.update(_tokenise(text))
    for text in low:
        low_words.update(_tokenise(text))

    total_high = max(sum(high_words.values()), 1)
    total_low = max(sum(low_words.values()), 1)

    rows = []
    all_words = set(high_words) | set(low_words)
    for word in all_words:
        h = high_words[word] / total_high
        l = low_words[word] / total_low
        if h + l > 0:
            rows.append({"word": word, "high_freq": h, "low_freq": l,
                         "lift": round(h / max(l, 0.001), 2),
                         "high_count": high_words[word]})

    result = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return result[result["high_count"] >= 2].head(n)


def top_phrases_by_engagement(posts: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Two-word phrases (bigrams) most correlated with high engagement."""
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    threshold = df["engagement"].median()
    high = df[df["engagement"] >= threshold]["caption"].fillna("")
    low = df[df["engagement"] < threshold]["caption"].fillna("")

    high_bg = Counter()
    low_bg = Counter()
    for text in high:
        high_bg.update(_bigrams(_tokenise(text)))
    for text in low:
        low_bg.update(_bigrams(_tokenise(text)))

    total_high = max(sum(high_bg.values()), 1)
    total_low = max(sum(low_bg.values()), 1)

    rows = []
    for phrase in set(high_bg):
        if high_bg[phrase] < 2:
            continue
        h = high_bg[phrase] / total_high
        l = low_bg[phrase] / total_low
        rows.append({"phrase": phrase, "lift": round(h / max(l, 0.001), 2),
                     "count": high_bg[phrase]})

    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(n)


def caption_length_buckets(posts: pd.DataFrame) -> pd.DataFrame:
    """Avg engagement by caption length bucket."""
    df = caption_word_stats(posts.copy())
    df["engagement"] = df["likes"] + df["comments"]
    df["length_bucket"] = pd.cut(
        df["word_count"],
        bins=[0, 10, 30, 60, 100, 10000],
        labels=["Micro (1-10)", "Short (11-30)", "Medium (31-60)", "Long (61-100)", "Essay (100+)"]
    )
    result = (
        df.groupby("length_bucket")["engagement"]
        .agg(["mean", "count"])
        .reset_index()
    )
    result.columns = ["length_bucket", "avg_engagement", "post_count"]
    result["avg_engagement"] = result["avg_engagement"].round(0)
    return result


def caption_feature_impact(posts: pd.DataFrame) -> pd.DataFrame:
    """Avg engagement for posts with/without questions, CTAs, emojis."""
    df = caption_word_stats(posts.copy())
    df["engagement"] = df["likes"] + df["comments"]
    base = df["engagement"].mean()

    rows = []
    for feature, label in [("has_question", "Ends with Question?"),
                            ("has_cta", "Has CTA (save/comment/etc)?"),
                            ("has_emoji", "Has Emoji?")]:
        with_feat = df[df[feature]]["engagement"].mean()
        without_feat = df[~df[feature]]["engagement"].mean()
        rows.append({
            "feature": label,
            "with_avg": round(with_feat, 0) if not pd.isna(with_feat) else 0,
            "without_avg": round(without_feat, 0) if not pd.isna(without_feat) else 0,
            "lift_pct": round((with_feat - without_feat) / max(without_feat, 1) * 100, 1) if not pd.isna(with_feat) else 0,
            "posts_with": int(df[feature].sum()),
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────

def word_lift_chart(word_df: pd.DataFrame) -> go.Figure:
    top = word_df.head(15).sort_values("lift")
    fig = go.Figure(go.Bar(
        x=top["lift"], y=top["word"], orientation="h",
        marker_color="#E1306C",
    ))
    fig.update_layout(
        template="plotly_dark", title="Top Words in High-Engagement Captions (Lift Score)",
        xaxis_title="Lift vs Low-Engagement Posts", yaxis_title="",
        height=420,
    )
    return fig


def caption_length_chart(bucket_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=bucket_df["length_bucket"].astype(str),
        y=bucket_df["avg_engagement"],
        marker_color="#833AB4",
        text=bucket_df["post_count"].apply(lambda n: f"{n} posts"),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", title="Average Engagement by Caption Length",
        xaxis_title="Caption Length", yaxis_title="Avg Engagement",
        height=380,
    )
    return fig


def feature_impact_chart(feat_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="With Feature", x=feat_df["feature"], y=feat_df["with_avg"],
        marker_color="#E1306C",
    ))
    fig.add_trace(go.Bar(
        name="Without Feature", x=feat_df["feature"], y=feat_df["without_avg"],
        marker_color="#555",
    ))
    fig.update_layout(
        template="plotly_dark", barmode="group",
        title="Caption Feature Impact on Engagement",
        yaxis_title="Avg Engagement", height=380,
    )
    return fig
