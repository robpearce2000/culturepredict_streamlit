"""
Posting schedule generator.
Builds an optimal weekly schedule from the creator's best timing data
and generates specific post ideas for each slot.
"""
import anthropic
import pandas as pd
from config import CLAUDE_MODEL, AI_MAX_TOKENS_IDEAS


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def best_slots(posts: pd.DataFrame, n_slots: int = 5) -> list[dict]:
    """Return the N best day+hour combinations by avg engagement."""
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    df["day"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour

    if df.empty:
        return []

    grouped = (
        df.groupby(["day", "hour"])["engagement"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Weight by both engagement and number of posts (need at least 1 data point)
    grouped = grouped[grouped["count"] >= 1]
    grouped = grouped.nlargest(n_slots, "mean")

    slots = []
    for _, row in grouped.iterrows():
        hour = int(row["hour"])
        slots.append({
            "day": row["day"],
            "hour": hour,
            "time": f"{hour:02d}:00",
            "avg_engagement": round(row["mean"], 0),
            "sample_posts": int(row["count"]),
        })
    # Sort by day of week order
    slots.sort(key=lambda s: DAYS.index(s["day"]) if s["day"] in DAYS else 7)
    return slots


def generate_weekly_schedule(
    posts: pd.DataFrame,
    competitor_posts: pd.DataFrame,
    api_key: str,
    niche: str = "",
    posts_per_week: int = 5,
) -> str:
    slots = best_slots(posts, n_slots=posts_per_week)

    if not slots:
        slots_text = "No timing data available — use these default high-performing times: Mon 18:00, Wed 12:00, Fri 18:00, Sat 10:00, Sun 19:00"
    else:
        slots_text = "\n".join(
            f"- {s['day']} at {s['time']} (avg engagement: {s['avg_engagement']:.0f})"
            for s in slots
        )

    best_type = "Reel"
    if not posts.empty:
        posts["engagement"] = posts["likes"] + posts["comments"]
        best_type = posts.groupby("content_type")["engagement"].mean().idxmax()

    comp_insight = ""
    if not competitor_posts.empty:
        competitor_posts["engagement"] = competitor_posts["likes"] + competitor_posts["comments"]
        comp_best = competitor_posts.groupby("content_type")["engagement"].mean().idxmax()
        comp_insight = f"Competitor's best performing format: {comp_best}"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_IDEAS,
        messages=[{
            "role": "user",
            "content": f"""You are an Instagram growth strategist. Create a specific weekly content schedule.

Data:
- Best posting slots (from their actual performance data):
{slots_text}
- Their best performing content type: {best_type}
- {comp_insight}
{f"- Niche: {niche}" if niche else ""}
- Posts per week: {posts_per_week}

Output a weekly schedule in this exact format for each slot:

**[Day] [Time]** — [Content Type]
Hook: [exact opening line to use]
Topic: [specific topic angle]
Format notes: [duration if Reel, slide count if carousel, etc.]

Make every hook specific and scroll-stopping. Reference the niche if provided.
End with one line: "💡 Pro tip: [one scheduling insight from the data]" """
        }]
    )
    return message.content[0].text
