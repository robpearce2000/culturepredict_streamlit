"""
Post scoring — gives every post a 1-10 performance score relative to the
creator's own average. Makes it instantly obvious which posts won and which flopped.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def score_posts(posts: pd.DataFrame, follower_count: int = 0) -> pd.DataFrame:
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]

    # Engagement rate if follower count provided
    if follower_count > 0:
        df["eng_rate_pct"] = (df["engagement"] / follower_count * 100).round(2)
    else:
        df["eng_rate_pct"] = None

    # Score 1-10 based on percentile within their own posts
    df["percentile"] = df["engagement"].rank(pct=True)
    df["score"] = (df["percentile"] * 9 + 1).round(1).clip(1, 10)

    # Label
    def label(s):
        if s >= 9: return "🔥 Viral"
        if s >= 7: return "✅ Strong"
        if s >= 5: return "👍 Average"
        if s >= 3: return "📉 Weak"
        return "❌ Flopped"

    df["verdict"] = df["score"].apply(label)
    return df


def score_distribution_chart(scored: pd.DataFrame) -> go.Figure:
    bins = pd.cut(scored["score"], bins=[0,2,4,6,8,10],
                  labels=["1-2 Flopped","3-4 Weak","5-6 Average","7-8 Strong","9-10 Viral"])
    counts = bins.value_counts().sort_index()
    colors = ["#333","#555","#833AB4","#E1306C","#ff6b6b"]
    fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values,
                           marker_color=colors))
    fig.update_layout(template="plotly_dark", title="Post Score Distribution",
                      xaxis_title="Score Band", yaxis_title="Number of Posts")
    return fig


def engagement_rate_chart(scored: pd.DataFrame) -> go.Figure:
    df = scored.dropna(subset=["eng_rate_pct"]).copy()
    if df.empty:
        return go.Figure().update_layout(title="Add follower count in Settings to see engagement rate", template="plotly_dark")
    df = df.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eng_rate_pct"],
                             mode="markers+lines", name="ER%",
                             marker=dict(color=df["score"], colorscale="RdPu", size=8),
                             line=dict(color="#444", width=1)))
    avg = df["eng_rate_pct"].mean()
    fig.add_hline(y=avg, line_dash="dash", line_color="#E1306C",
                  annotation_text=f"Avg {avg:.2f}%")
    fig.update_layout(template="plotly_dark", title="Engagement Rate % Over Time",
                      yaxis_title="Engagement Rate %", xaxis_title="Date")
    return fig


def top_and_bottom(scored: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["timestamp","content_type","caption","likes","comments","engagement","score","verdict"]
    top = scored.nlargest(n, "score")[cols].reset_index(drop=True)
    bottom = scored.nsmallest(n, "score")[cols].reset_index(drop=True)
    top["caption"] = top["caption"].str[:70] + "..."
    bottom["caption"] = bottom["caption"].str[:70] + "..."
    return top, bottom
