import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def engagement_over_time(posts: pd.DataFrame) -> go.Figure:
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp")
    fig = px.scatter(
        df, x="timestamp", y="engagement",
        color="content_type", size="engagement",
        hover_data=["caption", "likes", "comments"],
        title="Engagement Per Post Over Time",
        labels={"engagement": "Likes + Comments", "timestamp": "Date"},
        template="plotly_dark",
    )
    # rolling average line
    daily = df.groupby("date")["engagement"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    fig.add_scatter(x=daily["date"], y=daily["engagement"].rolling(7, min_periods=1).mean(),
                    mode="lines", name="7-day avg", line=dict(color="white", width=2))
    return fig


def content_type_breakdown(posts: pd.DataFrame) -> go.Figure:
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    grouped = df.groupby("content_type").agg(
        count=("engagement", "count"),
        avg_engagement=("engagement", "mean"),
        total_likes=("likes", "sum"),
    ).reset_index()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Post Count by Type", "Avg Engagement by Type"))
    fig.add_trace(go.Bar(x=grouped["content_type"], y=grouped["count"],
                         name="Count", marker_color="#E1306C"), row=1, col=1)
    fig.add_trace(go.Bar(x=grouped["content_type"], y=grouped["avg_engagement"].round(1),
                         name="Avg Engagement", marker_color="#833AB4"), row=1, col=2)
    fig.update_layout(template="plotly_dark", title="Content Type Performance", showlegend=False)
    return fig


def best_posting_times(posts: pd.DataFrame) -> go.Figure:
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    heatmap = df.groupby(["day", "hour"])["engagement"].mean().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap["day"] = pd.Categorical(heatmap["day"], categories=day_order, ordered=True)
    heatmap = heatmap.sort_values("day")
    pivot = heatmap.pivot(index="day", columns="hour", values="engagement").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdPu",
        hoverongaps=False,
    ))
    fig.update_layout(template="plotly_dark", title="Best Posting Times (Avg Engagement)",
                      xaxis_title="Hour of Day", yaxis_title="Day of Week")
    return fig


def follower_growth(followers: pd.DataFrame) -> go.Figure:
    if followers.empty:
        return go.Figure().update_layout(title="No follower data found", template="plotly_dark")
    df = followers.dropna(subset=["timestamp"]).copy()
    df["date"] = df["timestamp"].dt.date
    df = df.groupby("date").size().reset_index(name="new_followers")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["total"] = df["new_followers"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["total"], fill="tozeroy",
                             name="Total Followers", line=dict(color="#E1306C")))
    fig.update_layout(template="plotly_dark", title="Follower Growth Over Time",
                      xaxis_title="Date", yaxis_title="Cumulative Followers")
    return fig


def top_posts_table(posts: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    df["engagement_rate"] = (df["engagement"] / df["engagement"].max() * 100).round(1)
    top = df.nlargest(10, "engagement")[
        ["timestamp", "content_type", "caption", "likes", "comments", "engagement"]
    ].reset_index(drop=True)
    top["caption"] = top["caption"].str[:80] + "..."
    return top


def caption_length_vs_engagement(posts: pd.DataFrame) -> go.Figure:
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    df["caption_len"] = df["caption"].fillna("").str.len()
    df["caption_bucket"] = pd.cut(df["caption_len"],
                                   bins=[0, 50, 150, 300, 1000, 10000],
                                   labels=["<50", "50-150", "150-300", "300-1000", "1000+"])
    grouped = df.groupby("caption_bucket", observed=True)["engagement"].mean().reset_index()
    fig = px.bar(grouped, x="caption_bucket", y="engagement",
                 title="Caption Length vs Avg Engagement",
                 labels={"caption_bucket": "Caption Length (chars)", "engagement": "Avg Engagement"},
                 color="engagement", color_continuous_scale="RdPu",
                 template="plotly_dark")
    return fig


def hashtag_performance(posts: pd.DataFrame) -> go.Figure:
    import re
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    df["hashtags"] = df["caption"].fillna("").apply(lambda c: re.findall(r"#\w+", c.lower()))
    exploded = df.explode("hashtags").dropna(subset=["hashtags"])
    if exploded.empty:
        return go.Figure().update_layout(title="No hashtags found in captions", template="plotly_dark")
    tag_perf = exploded.groupby("hashtags")["engagement"].mean().reset_index()
    tag_perf = tag_perf[tag_perf["hashtags"].str.len() > 1].nlargest(20, "engagement")
    fig = px.bar(tag_perf, x="engagement", y="hashtags", orientation="h",
                 title="Top 20 Hashtags by Avg Engagement",
                 color="engagement", color_continuous_scale="RdPu",
                 template="plotly_dark")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
