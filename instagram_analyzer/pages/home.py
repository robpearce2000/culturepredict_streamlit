import streamlit as st
import anthropic
import pandas as pd
from datetime import datetime

from cache import load_analysis, cache_age_str, list_cached_competitors, load_competitor
from prompts import what_to_post_today
from config import CLAUDE_MODEL, AI_MAX_TOKENS_CAPTION


def _days_since_last_post(posts: pd.DataFrame) -> int:
    if posts.empty or posts["timestamp"].isna().all():
        return 0
    latest = posts["timestamp"].max()
    return (datetime.now() - latest).days


def _worst_content_type(posts: pd.DataFrame) -> str:
    if posts.empty:
        return "unknown"
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    by_type = df.groupby("content_type")["engagement"].mean()
    return by_type.idxmin() if not by_type.empty else "unknown"


def _best_competitor_type(competitor_posts: pd.DataFrame) -> str:
    if competitor_posts.empty:
        return "unknown"
    df = competitor_posts.copy()
    by_type = df.groupby("content_type")["engagement"].mean()
    return by_type.idxmax() if not by_type.empty else "unknown"


def _get_today_recommendation(posts, competitor_posts, api_key) -> str:
    import re
    df = posts.copy()
    df["engagement"] = df["likes"] + df["comments"]
    df["tags"] = df["caption"].fillna("").apply(lambda c: re.findall(r"#\w+", c.lower()))
    exploded = df.explode("tags").dropna(subset=["tags"])
    top_tags = (
        exploded.groupby("tags")["engagement"].mean().nlargest(5).index.tolist()
        if not exploded.empty else []
    )

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_CAPTION,
        messages=[{
            "role": "user",
            "content": what_to_post_today(
                worst_type=_worst_content_type(posts),
                best_competitor_type=_best_competitor_type(competitor_posts),
                days_since_post=_days_since_last_post(posts),
                top_hashtags=", ".join(top_tags) if top_tags else "N/A",
                avg_engagement=df["engagement"].mean() if not df.empty else 0,
            ),
        }],
    )
    return msg.content[0].text


def render():
    st.title("🏠 Home")

    cached = load_analysis()
    competitors = list_cached_competitors()
    comp_posts = pd.DataFrame()
    if competitors:
        comp_data = load_competitor(competitors[0])
        if comp_data:
            comp_posts = comp_data["posts"]

    if not cached:
        st.info("No data yet. Go to **My Analytics** and upload your Instagram export ZIP to get started.")
        st.markdown("""
### What this tool does
- **My Analytics** — upload your Instagram export and see exactly what's working
- **Competitor Spy** — enter any public @handle and analyse their content strategy
- **Video Formulas** — AI extracts the exact Reel structure your competitors use
- **Content Ideas** — AI generates 10 post ideas based on what's performing in your niche
- **Hashtag Strategy** — find the hashtags your competitors use that you don't
""")
        return

    posts = cached["posts"]
    profile = cached["profile"]
    age = cache_age_str()

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.caption(f"Data from your Instagram export · {age}")

    posts["engagement"] = posts["likes"] + posts["comments"]
    avg_eng = posts["engagement"].mean()
    best_type = posts.groupby("content_type")["engagement"].mean().idxmax() if not posts.empty else "N/A"
    days_ago = _days_since_last_post(posts)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts Analysed", len(posts))
    c2.metric("Avg Engagement", f"{avg_eng:.0f}")
    c3.metric("Best Format", best_type)
    c4.metric("Days Since Last Post", days_ago, delta=f"-{days_ago}d" if days_ago > 3 else "✓ recent", delta_color="inverse")
    c5.metric("Competitors Tracked", len(competitors))

    st.markdown("---")

    # ── What to post today ────────────────────────────────────────────────────
    st.subheader("📌 What to Post Today")

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings to get your daily recommendation.")
    else:
        cache_key = "today_recommendation"
        col_a, col_b = st.columns([6, 1])
        with col_b:
            if st.button("🔄 Refresh", help="Generate a new recommendation"):
                if cache_key in st.session_state:
                    del st.session_state[cache_key]

        if cache_key not in st.session_state:
            with st.spinner("Generating your recommendation..."):
                try:
                    st.session_state[cache_key] = _get_today_recommendation(posts, comp_posts, api_key)
                except Exception as e:
                    st.error(f"Could not generate recommendation: {e}")

        if cache_key in st.session_state:
            st.markdown(
                f"""<div style="background:#1a1a2e;border-left:4px solid #E1306C;padding:16px 20px;border-radius:6px;margin-bottom:8px">
{st.session_state[cache_key]}
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Quick stats ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Top 5 Posts")
        top5 = posts.nlargest(5, "engagement")[
            ["timestamp", "content_type", "likes", "comments", "engagement"]
        ].reset_index(drop=True)
        top5["timestamp"] = top5["timestamp"].dt.strftime("%d %b %Y")
        st.dataframe(top5, use_container_width=True)

    with col2:
        if competitors:
            st.subheader(f"Latest Competitor: {competitors[0]}")
            if not comp_posts.empty:
                comp_top5 = comp_posts.nlargest(5, "engagement")[
                    ["timestamp", "content_type", "likes", "comments", "engagement"]
                ].reset_index(drop=True)
                comp_top5["timestamp"] = comp_top5["timestamp"].dt.strftime("%d %b %Y")
                st.dataframe(comp_top5, use_container_width=True)
        else:
            st.subheader("No Competitors Tracked Yet")
            st.info("Go to **Competitor Spy** and enter a @handle to start tracking.")
