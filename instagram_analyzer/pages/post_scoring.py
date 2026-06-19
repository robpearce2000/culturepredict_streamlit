import streamlit as st
from cache import load_analysis
from scoring import score_posts, score_distribution_chart, engagement_rate_chart, top_and_bottom


def render():
    st.title("🏆 Post Scoring")
    st.caption("Every post scored 1-10 relative to your own average — instantly see what won and what flopped.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return

    posts = cached["posts"]
    if posts.empty:
        st.warning("No post data found.")
        return

    follower_count = st.session_state.get("follower_count", 0)

    scored = score_posts(posts, follower_count=follower_count)

    # ── Summary metrics ───────────────────────────────────────────────────────
    viral = (scored["score"] >= 9).sum()
    strong = ((scored["score"] >= 7) & (scored["score"] < 9)).sum()
    flopped = (scored["score"] < 3).sum()
    avg_score = scored["score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Score", f"{avg_score:.1f}/10")
    c2.metric("Viral Posts (9-10)", int(viral))
    c3.metric("Strong Posts (7-8)", int(strong))
    c4.metric("Flopped (<3)", int(flopped))

    if follower_count == 0:
        st.info("Add your follower count in ⚙️ Settings to unlock Engagement Rate % column.")

    st.markdown("---")

    # ── Score distribution ────────────────────────────────────────────────────
    st.plotly_chart(score_distribution_chart(scored), use_container_width=True)

    if follower_count > 0:
        st.plotly_chart(engagement_rate_chart(scored), use_container_width=True)

    st.markdown("---")

    # ── Top and bottom posts ──────────────────────────────────────────────────
    top, bottom = top_and_bottom(scored)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Posts")
        st.dataframe(top, use_container_width=True)
    with col2:
        st.subheader("Bottom 5 Posts")
        st.dataframe(bottom, use_container_width=True)

    st.markdown("---")

    # ── Full scored table ─────────────────────────────────────────────────────
    with st.expander("All Posts (sorted by score)"):
        display_cols = ["timestamp", "content_type", "score", "verdict", "likes", "comments", "engagement"]
        if follower_count > 0:
            display_cols.append("eng_rate_pct")
        display = scored[display_cols].sort_values("score", ascending=False).reset_index(drop=True)
        st.dataframe(display, use_container_width=True)
