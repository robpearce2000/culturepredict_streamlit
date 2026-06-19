import anthropic
import pandas as pd

from config import CLAUDE_MODEL, AI_MAX_TOKENS_FORMULA, AI_MAX_TOKENS_SUMMARY
from prompts import video_formula, content_mix_analysis


def extract_video_formula(posts: pd.DataFrame, handle: str, api_key: str) -> str:
    df = posts[posts["content_type"].isin(["Reel", "Video"])].copy()
    if df.empty:
        return "No video content found for this account."

    df = df.nlargest(10, "engagement")
    summaries = []
    for _, row in df.iterrows():
        duration = f"{int(row['video_duration'])}s" if pd.notna(row.get("video_duration")) else "unknown"
        caption = str(row.get("caption", ""))[:200]
        summaries.append(
            f"- Engagement: {row['engagement']:,} | Duration: {duration} | Caption: {caption}"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_FORMULA,
        messages=[{
            "role": "user",
            "content": video_formula(handle, "\n".join(summaries), len(df)),
        }],
    )
    return message.content[0].text


def analyse_content_mix(posts: pd.DataFrame, handle: str, api_key: str) -> str:
    if posts.empty:
        return "No posts to analyse."

    type_perf = posts.groupby("content_type")["engagement"].agg(["mean", "count"]).round(0)
    top3 = posts.nlargest(3, "engagement")[["content_type", "caption", "engagement", "hashtag_count"]]

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_SUMMARY,
        messages=[{
            "role": "user",
            "content": content_mix_analysis(handle, type_perf.to_string(), top3.to_string(index=False)),
        }],
    )
    return message.content[0].text
