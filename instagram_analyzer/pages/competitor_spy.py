import streamlit as st
import plotly.express as px
import pandas as pd

from competitor import scrape_profile, engagement_rate_df
from cache import save_competitor, load_competitor, list_cached_competitors
from config import COMPETITOR_MAX_POSTS


def _competitor_charts(posts: pd.DataFrame, handle: str):
    posts = posts.copy()

    col1, col2 = st.columns(2)
    with col1:
        type_counts = posts["content_type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig = px.bar(type_counts, x="type", y="count", title="Post Mix",
                     color="type", template="plotly_dark",
                     color_discrete_sequence=["#E1306C", "#833AB4", "#F77737", "#FCAF45"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        type_eng = posts.groupby("content_type")["engagement"].mean().reset_index()
        fig2 = px.bar(type_eng, x="content_type", y="engagement",
                      title="Avg Engagement by Type",
                      color="engagement", color_continuous_scale="RdPu",
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # Top posts
    st.subheader(f"@{handle}'s Top Posts")
    top = posts.nlargest(10, "engagement")[
        ["timestamp", "content_type", "likes", "comments", "engagement", "caption"]
    ].reset_index(drop=True)
    top["caption"] = top["caption"].str[:80] + "..."
    top["timestamp"] = pd.to_datetime(top["timestamp"]).dt.strftime("%d %b %Y")
    st.dataframe(top, use_container_width=True)

    # Posting frequency
    if posts["timestamp"].notna().any():
        posts["date"] = pd.to_datetime(posts["timestamp"]).dt.date
        freq = posts.groupby("date").size().reset_index(name="posts")
        freq["date"] = pd.to_datetime(freq["date"])
        fig3 = px.bar(freq, x="date", y="posts", title="Posting Frequency",
                      template="plotly_dark", color_discrete_sequence=["#E1306C"])
        st.plotly_chart(fig3, use_container_width=True)


def render():
    st.title("🔍 Competitor Spy")

    cached_handles = list_cached_competitors()

    # ── Handle input ──────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        handle = st.text_input("Instagram @handle to analyse", placeholder="@handle or handle")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        scrape = st.button("Analyse", type="primary")

    if handle and scrape:
        handle = handle.lstrip("@")
        with st.spinner(f"Pulling @{handle}'s last {COMPETITOR_MAX_POSTS} posts..."):
            try:
                posts = scrape_profile(handle, COMPETITOR_MAX_POSTS)
                if posts.empty:
                    st.error("No posts found. Check the handle is correct and the account is public.")
                else:
                    save_competitor(handle, posts)
                    st.success(f"Pulled {len(posts)} posts from @{handle}")
                    st.session_state["active_competitor"] = handle
            except Exception as e:
                st.error(f"Scrape failed: {e}")
                st.caption("If you haven't set an Apify API token, the app uses demo data. Add APIFY_API_TOKEN to your .env file for live data.")

    # ── Previously tracked ────────────────────────────────────────────────────
    if cached_handles:
        st.markdown("---")
        selected = st.selectbox(
            "Previously analysed accounts",
            cached_handles,
            index=0 if "active_competitor" not in st.session_state
                  else (cached_handles.index(f"@{st.session_state.get('active_competitor', '')}") if f"@{st.session_state.get('active_competitor','')}" in cached_handles else 0)
        )
        active = selected.lstrip("@")
        data = load_competitor(active)

        if data:
            posts = data["posts"]
            posts["engagement"] = posts["likes"] + posts["comments"]
            st.caption(f"@{active} · {len(posts)} posts · cached data")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Posts Analysed", len(posts))
            m2.metric("Avg Engagement", f"{posts['engagement'].mean():.0f}")
            m3.metric("Best Format", posts.groupby("content_type")["engagement"].mean().idxmax())
            m4.metric("Avg Hashtags", f"{posts['hashtag_count'].mean():.1f}")

            st.markdown("---")
            _competitor_charts(posts, active)
    else:
        if not handle:
            st.info("Enter a competitor's @handle above to start. Uses demo data if no Apify token is set.")
