import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def monthly_stats(posts: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    df["month"] = df["timestamp"].dt.to_period("M")
    return (
        df.groupby("month")
        .agg(
            posts=("engagement", "count"),
            avg_engagement=("engagement", "mean"),
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
        )
        .reset_index()
    )
    for col in ["avg_engagement", "total_likes", "total_comments"]:
        monthly[col] = monthly[col].round(1)
    return monthly


def growth_velocity(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    df["engagement_change_pct"] = df["avg_engagement"].pct_change() * 100
    df["posts_change"] = df["posts"].diff()
    return df.round(1)


def trend_chart(posts: pd.DataFrame) -> go.Figure:
    monthly = monthly_stats(posts)
    if len(monthly) < 2:
        return go.Figure().update_layout(title="Not enough data for trends (need 2+ months)", template="plotly_dark")

    vel = growth_velocity(monthly)
    month_labels = [str(m) for m in vel["month"]]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Avg Engagement per Post", "Posts per Month", "Month-on-Month Engagement Change %", "Total Likes + Comments"),
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(x=month_labels, y=vel["avg_engagement"], mode="lines+markers",
                             name="Avg Engagement", line=dict(color="#E1306C", width=2)), row=1, col=1)

    fig.add_trace(go.Bar(x=month_labels, y=vel["posts"], name="Posts",
                         marker_color="#833AB4"), row=1, col=2)

    colors = ["#E1306C" if v >= 0 else "#555" for v in vel["engagement_change_pct"].fillna(0)]
    fig.add_trace(go.Bar(x=month_labels, y=vel["engagement_change_pct"],
                         name="Change %", marker_color=colors), row=2, col=1)

    total_eng = vel["total_likes"] + vel["total_comments"]
    fig.add_trace(go.Scatter(x=month_labels, y=total_eng, fill="tozeroy", mode="lines",
                             name="Total Engagement", line=dict(color="#F77737")), row=2, col=2)

    fig.update_layout(template="plotly_dark", title="Growth Trends", showlegend=False, height=600)
    return fig


def content_type_trend(posts: pd.DataFrame) -> go.Figure:
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    grouped = df.groupby(["month", "content_type"])["engagement"].mean().reset_index()
    fig = px.line(grouped, x="month", y="engagement", color="content_type",
                  title="Avg Engagement by Content Type Over Time",
                  template="plotly_dark", markers=True,
                  color_discrete_sequence=["#E1306C", "#833AB4", "#F77737", "#FCAF45"])
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Engagement")
    return fig


def best_and_worst_months(monthly: pd.DataFrame) -> tuple[dict, dict]:
    if monthly.empty:
        return {}, {}
    best = monthly.loc[monthly["avg_engagement"].idxmax()].to_dict()
    worst = monthly.loc[monthly["avg_engagement"].idxmin()].to_dict()
    return best, worst
