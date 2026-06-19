import anthropic
import pandas as pd
import re

from config import CLAUDE_MODEL, AI_MAX_TOKENS_SUMMARY, TOP_HASHTAGS_COUNT
from prompts import growth_summary


def build_stats_snapshot(posts: pd.DataFrame) -> str:
    if posts.empty:
        return "No post data available."

    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]

    total_posts = len(df)
    avg_eng = df["engagement"].mean()
    top5 = df.nlargest(5, "engagement")[["content_type", "caption", "likes", "comments"]]
    type_stats = df.groupby("content_type")["engagement"].agg(["mean", "count"]).round(1)

    best_hour = None
    if df["timestamp"].notna().any():
        df["hour"] = df["timestamp"].dt.hour
        best_hour = df.groupby("hour")["engagement"].mean().idxmax()

    df["hashtags"] = df["caption"].fillna("").apply(lambda c: re.findall(r"#\w+", c.lower()))
    exploded = df.explode("hashtags").dropna(subset=["hashtags"])
    top_tags = []
    if not exploded.empty:
        top_tags = (
            exploded.groupby("hashtags")["engagement"]
            .mean()
            .nlargest(TOP_HASHTAGS_COUNT)
            .head(5)
            .index.tolist()
        )

    return f"""Total posts analysed: {total_posts}
Average engagement (likes + comments): {avg_eng:.1f}

Performance by content type:
{type_stats.to_string()}

Top 5 posts by engagement:
{top5.to_string(index=False)}

Best posting hour: {best_hour}:00
Top performing hashtags: {", ".join(top_tags) if top_tags else "N/A"}"""


def get_ai_summary(posts: pd.DataFrame, api_key: str) -> str:
    snapshot = build_stats_snapshot(posts)
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_SUMMARY,
        messages=[{"role": "user", "content": growth_summary(snapshot)}],
    )
    return message.content[0].text
