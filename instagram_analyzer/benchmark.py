import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def head_to_head_metrics(my_posts: pd.DataFrame, comp_posts: pd.DataFrame, comp_handle: str) -> pd.DataFrame:
    def summarise(posts, label):
        posts = posts.copy()
        posts["engagement"] = posts["likes"] + posts["comments"]
        return {
            "Account": label,
            "Avg Engagement": round(posts["engagement"].mean(), 1),
            "Avg Likes": round(posts["likes"].mean(), 1),
            "Avg Comments": round(posts["comments"].mean(), 1),
            "Top Post Engagement": int(posts["engagement"].max()),
            "Posts Analysed": len(posts),
            "Avg Hashtags": round(posts.get("hashtag_count", pd.Series([0])).mean(), 1) if "hashtag_count" in posts.columns else "N/A",
            "Best Content Type": posts.groupby("content_type")["engagement"].mean().idxmax(),
        }

    return pd.DataFrame([summarise(my_posts, "You"), summarise(comp_posts, f"@{comp_handle}")])


def head_to_head_chart(my_posts: pd.DataFrame, comp_posts: pd.DataFrame, comp_handle: str) -> go.Figure:
    def stats(posts):
        posts = posts.copy()
        posts["engagement"] = posts["likes"] + posts["comments"]
        by_type = posts.groupby("content_type")["engagement"].mean().reset_index()
        return by_type

    my_stats = stats(my_posts)
    comp_stats = stats(comp_posts)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Avg Engagement by Content Type — You", f"Avg Engagement by Content Type — @{comp_handle}"))
    fig.add_trace(go.Bar(x=my_stats["content_type"], y=my_stats["engagement"],
                         marker_color="#E1306C", name="You"), row=1, col=1)
    fig.add_trace(go.Bar(x=comp_stats["content_type"], y=comp_stats["engagement"],
                         marker_color="#833AB4", name=f"@{comp_handle}"), row=1, col=2)
    fig.update_layout(template="plotly_dark", showlegend=False, title="Head-to-Head: Content Type Performance")
    return fig


def posting_frequency_comparison(my_posts: pd.DataFrame, comp_posts: pd.DataFrame, comp_handle: str) -> go.Figure:
    def freq(posts, label):
        df = posts.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        monthly = df.groupby("month").size().reset_index(name="posts")
        monthly["account"] = label
        return monthly

    my_freq = freq(my_posts, "You")
    comp_freq = freq(comp_posts, f"@{comp_handle}")
    combined = pd.concat([my_freq, comp_freq])

    fig = px.line(combined, x="month", y="posts", color="account",
                  title="Posting Frequency Comparison",
                  template="plotly_dark", markers=True,
                  color_discrete_map={"You": "#E1306C", f"@{comp_handle}": "#833AB4"})
    return fig


def engagement_distribution(my_posts: pd.DataFrame, comp_posts: pd.DataFrame, comp_handle: str) -> go.Figure:
    my = my_posts.copy()
    comp = comp_posts.copy()
    my["engagement"] = my["likes"] + my["comments"]
    comp["engagement"] = comp["likes"] + comp["comments"]

    fig = go.Figure()
    fig.add_trace(go.Box(y=my["engagement"], name="You", marker_color="#E1306C",
                         boxmean=True))
    fig.add_trace(go.Box(y=comp["engagement"], name=f"@{comp_handle}", marker_color="#833AB4",
                         boxmean=True))
    fig.update_layout(template="plotly_dark", title="Engagement Distribution",
                      yaxis_title="Engagement per Post")
    return fig
