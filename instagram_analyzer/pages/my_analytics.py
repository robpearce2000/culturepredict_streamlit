import streamlit as st

from parser import load_export
from cache import save_analysis, load_analysis, cache_age_str
from charts import (
    engagement_over_time,
    content_type_breakdown,
    best_posting_times,
    follower_growth,
    top_posts_table,
    caption_length_vs_engagement,
    hashtag_performance,
)
from ai_summary import get_ai_summary


def render():
    st.title("📊 My Analytics")

    api_key = st.session_state.get("api_key", "")
    cached = load_analysis()

    # ── Upload section ────────────────────────────────────────────────────────
    with st.expander("📁 Upload / Refresh Instagram Export", expanded=not bool(cached)):
        st.markdown("""
Upload the ZIP file from **Instagram → Settings → Your activity → Download your information**.
Select JSON format, choose Posts + Reels + Stories + Followers.
        """)
        uploaded = st.file_uploader("Instagram data export ZIP", type=["zip"])
        if uploaded:
            with st.spinner("Parsing your export..."):
                try:
                    data = load_export(uploaded)
                    save_analysis(data["posts"], data["followers"], data["profile"])
                    cached = load_analysis()
                    st.success(f"Loaded {len(data['posts'])} posts and {len(data['followers'])} followers.")
                    if data.get("file_list"):
                        with st.expander("Files found in ZIP (debug)"):
                            st.code("\n".join(data["file_list"][:40]))
                except Exception as e:
                    st.error(f"Could not parse export: {e}")
                    st.caption("Make sure you selected JSON format (not HTML) when downloading from Instagram.")

    if not cached:
        st.info("Upload your Instagram export ZIP above to see your analytics.")
        return

    posts = cached["posts"]
    followers = cached["followers"]
    profile = cached["profile"]
    st.caption(f"@{profile['username']} · {cache_age_str()}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    posts["engagement"] = posts["likes"] + posts["comments"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analysed", len(posts))
    c2.metric("Avg Engagement", f"{posts['engagement'].mean():.0f}")
    c3.metric("Total Followers (export)", profile["followers"])
    c4.metric("Following", profile["following"])

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Content Strategy", "Timing", "AI Insights"])

    with tab1:
        st.plotly_chart(engagement_over_time(posts), use_container_width=True)
        st.plotly_chart(follower_growth(followers), use_container_width=True)
        st.subheader("Top 10 Posts")
        st.dataframe(top_posts_table(posts), use_container_width=True)

    with tab2:
        st.plotly_chart(content_type_breakdown(posts), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(caption_length_vs_engagement(posts), use_container_width=True)
        with col_b:
            st.plotly_chart(hashtag_performance(posts), use_container_width=True)

    with tab3:
        st.plotly_chart(best_posting_times(posts), use_container_width=True)
        st.caption("Darker = higher avg engagement at that day/hour combination.")

    with tab4:
        st.subheader("AI Growth Insights")
        if not api_key:
            st.warning("Add your Claude API key in ⚙️ Settings.")
        else:
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("Analysing your data..."):
                    try:
                        summary = get_ai_summary(posts, api_key)
                        st.session_state["my_ai_summary"] = summary
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
            if "my_ai_summary" in st.session_state:
                st.markdown(st.session_state["my_ai_summary"])
