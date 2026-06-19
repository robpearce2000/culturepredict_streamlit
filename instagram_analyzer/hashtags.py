"""
Hashtag strategy analyser.
Combines your own hashtag performance with competitor hashtag data.
"""
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def my_hashtag_performance(posts: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    df["tags"] = df["caption"].fillna("").apply(lambda c: re.findall(r"#\w+", c.lower()))
    exploded = df.explode("tags").dropna(subset=["tags"])
    exploded = exploded[exploded["tags"].str.len() > 2]
    if exploded.empty:
        return pd.DataFrame()
    return (
        exploded.groupby("tags")
        .agg(
            uses=("engagement", "count"),
            avg_engagement=("engagement", "mean"),
            total_engagement=("engagement", "sum"),
        )
        .round(1)
        .sort_values("avg_engagement", ascending=False)
        .reset_index()
        .rename(columns={"tags": "hashtag"})
    )


def competitor_hashtag_overlap(my_posts: pd.DataFrame, competitor_posts: pd.DataFrame) -> pd.DataFrame:
    def extract_tags(posts):
        tags = set()
        for caption in posts["caption"].fillna(""):
            tags.update(re.findall(r"#\w+", caption.lower()))
        return tags

    my_tags = extract_tags(my_posts) if not my_posts.empty else set()
    comp_tags = extract_tags(competitor_posts) if not competitor_posts.empty else set()

    rows = []
    for tag in comp_tags:
        rows.append({
            "hashtag": tag,
            "you_use": tag in my_tags,
            "competitor_uses": True,
        })
    df = pd.DataFrame(rows)
    df["opportunity"] = ~df["you_use"]
    return df.sort_values(["opportunity", "hashtag"], ascending=[False, True])


def hashtag_chart(perf: pd.DataFrame) -> go.Figure:
    if perf.empty:
        return go.Figure().update_layout(title="No hashtag data", template="plotly_dark")
    top = perf.head(20)
    fig = px.bar(
        top, x="avg_engagement", y="hashtag", orientation="h",
        title="Your Top 20 Hashtags by Avg Engagement",
        color="avg_engagement", color_continuous_scale="RdPu",
        template="plotly_dark",
        labels={"avg_engagement": "Avg Engagement", "hashtag": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def gap_chart(overlap: pd.DataFrame) -> go.Figure:
    if overlap.empty:
        return go.Figure().update_layout(title="No overlap data", template="plotly_dark")
    opportunities = overlap[overlap["opportunity"]].head(25)
    fig = px.bar(
        opportunities, x="hashtag", y=[1] * len(opportunities),
        title="Hashtags Competitors Use That You Don't",
        template="plotly_dark",
        labels={"y": "", "hashtag": ""},
        color_discrete_sequence=["#E1306C"],
    )
    fig.update_layout(yaxis_visible=False, showlegend=False)
    return fig
