"""
Viral score predictor — paste a post idea and AI scores it 1-10 with specific reasoning.
"""
import streamlit as st
import anthropic

from cache import load_analysis, list_cached_competitors, load_competitor
from config import CLAUDE_MODEL, AI_MAX_TOKENS_SUMMARY


def _viral_score_prompt(idea: str, format_type: str, account_context: str, comp_context: str) -> str:
    return f"""You are an Instagram growth expert. Score this post idea 1-10 for viral potential.

Post idea: "{idea}"
Format: {format_type}

{account_context}
{comp_context}

Output exactly this structure — nothing else:

**Viral Score: X/10**

**Why this score:**
[2 sentences referencing the account data and what works in this niche]

**What's strong:**
• [specific strength]
• [specific strength]

**What to improve:**
• [specific fix — be prescriptive]
• [specific fix — be prescriptive]

**Rewritten hook:** [a stronger opening line for this post]

**Best format for this idea:** [Reel / Carousel / Photo — and why based on the data]"""


def render():
    st.title("🎯 Viral Score Predictor")
    st.caption("Paste your post idea and get an AI score 1-10 before you waste time creating it.")

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings to use this feature.")
        return

    # Build context from cached data
    cached = load_analysis()
    account_context = ""
    if cached and not cached["posts"].empty:
        posts = cached["posts"].copy()
        posts["engagement"] = posts["likes"] + posts["comments"]
        best_type = posts.groupby("content_type")["engagement"].mean().idxmax()
        avg_eng = posts["engagement"].mean()
        top_posts = posts.nlargest(3, "engagement")["caption"].fillna("").tolist()
        top_captions = "\n".join(f"- {c[:100]}" for c in top_posts if c)
        account_context = f"""Account context:
- Best performing content type: {best_type}
- Average engagement: {avg_eng:.0f}
- Top performing captions (first 100 chars):
{top_captions}"""
    else:
        account_context = "Account context: No data uploaded yet (upload Instagram export for personalised scores)"

    comp_handles = list_cached_competitors()
    comp_context = ""
    if comp_handles:
        comp_data = load_competitor(comp_handles[0])
        if comp_data:
            comp = comp_data["posts"].copy()
            comp["engagement"] = comp["likes"] + comp["comments"]
            comp_best = comp.groupby("content_type")["engagement"].mean().idxmax()
            comp_top = comp.nlargest(3, "engagement")["caption"].fillna("").tolist()
            comp_captions = "\n".join(f"- {c[:100]}" for c in comp_top if c)
            comp_context = f"""Competitor context (@{comp_handles[0]}):
- Their best format: {comp_best}
- Their top captions:
{comp_captions}"""

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        idea = st.text_area(
            "Your post idea",
            placeholder="e.g. 'A reel showing 3 mistakes beginners make when learning guitar, ending with a quick fix for each'",
            height=100,
        )
    with col2:
        format_type = st.selectbox("Format", ["Reel", "Carousel", "Photo", "Story", "Not sure yet"])

    if st.button("Score My Idea", type="primary", disabled=not idea.strip()):
        with st.spinner("Scoring your idea..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=AI_MAX_TOKENS_SUMMARY,
                    messages=[{"role": "user", "content": _viral_score_prompt(
                        idea, format_type, account_context, comp_context
                    )}],
                )
                st.session_state["viral_score_result"] = msg.content[0].text
                st.session_state["viral_score_idea"] = idea
            except Exception as e:
                st.error(f"Scoring failed: {e}")

    if "viral_score_result" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["viral_score_result"])

        # Score history in session
        if "score_history" not in st.session_state:
            st.session_state["score_history"] = []

        # Parse score from result
        result_text = st.session_state["viral_score_result"]
        import re
        match = re.search(r"Viral Score:\s*(\d+)/10", result_text)
        if match:
            score = int(match.group(1))
            idea_short = st.session_state["viral_score_idea"][:60] + "..."
            # Add if not already in history
            existing = [h["idea"] for h in st.session_state["score_history"]]
            if idea_short not in existing:
                st.session_state["score_history"].append({"idea": idea_short, "score": score, "format": format_type})

    # Score history
    if st.session_state.get("score_history"):
        st.markdown("---")
        st.subheader("Score History (this session)")
        import pandas as pd
        hist_df = pd.DataFrame(st.session_state["score_history"])
        hist_df.columns = ["Idea", "Score", "Format"]
        hist_df = hist_df.sort_values("Score", ascending=False).reset_index(drop=True)
        st.dataframe(hist_df, use_container_width=True)

        if st.button("Clear history"):
            st.session_state["score_history"] = []
            st.rerun()
