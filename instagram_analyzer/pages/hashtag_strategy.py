import streamlit as st
import pandas as pd

from cache import load_analysis, list_cached_competitors, load_competitor
from hashtags import my_hashtag_performance, competitor_hashtag_overlap, hashtag_chart, gap_chart
from competitor import scrape_hashtag
from cache import save_competitor
from config import COMPETITOR_MAX_POSTS


def render():
    st.title("# Hashtag Strategy")
    st.caption("Find which hashtags actually drive engagement for you — and which ones your competitors use that you're missing.")

    my_data = load_analysis()
    my_posts = my_data["posts"] if my_data else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["My Hashtags", "Competitor Gap", "Hashtag Explorer"])

    with tab1:
        st.subheader("Your Hashtag Performance")
        if my_posts.empty:
            st.info("Upload your Instagram export in **My Analytics** to see your hashtag data.")
        else:
            perf = my_hashtag_performance(my_posts)
            if perf.empty:
                st.warning("No hashtags found in your captions.")
            else:
                st.plotly_chart(hashtag_chart(perf), use_container_width=True)
                st.subheader("Full Hashtag Table")
                st.dataframe(
                    perf.rename(columns={"hashtag": "Hashtag", "uses": "Times Used",
                                         "avg_engagement": "Avg Engagement", "total_engagement": "Total Engagement"}),
                    use_container_width=True,
                )

    with tab2:
        st.subheader("Hashtag Gap Analysis")
        competitors = list_cached_competitors()
        if not competitors:
            st.info("Analyse a competitor in **Competitor Spy** first.")
        elif my_posts.empty:
            st.info("Upload your Instagram export in **My Analytics** first.")
        else:
            handle = st.selectbox("Compare against", competitors).lstrip("@")
            comp_data = load_competitor(handle)
            if comp_data:
                comp_posts = comp_data["posts"]
                overlap = competitor_hashtag_overlap(my_posts, comp_posts)

                opportunities = overlap[overlap["opportunity"]]
                shared = overlap[~overlap["opportunity"]]

                col1, col2 = st.columns(2)
                col1.metric("Tags they use you don't", len(opportunities))
                col2.metric("Tags you both use", len(shared))

                st.plotly_chart(gap_chart(overlap), use_container_width=True)

                st.subheader("Opportunity Hashtags (they use, you don't)")
                st.dataframe(
                    opportunities[["hashtag"]].reset_index(drop=True),
                    use_container_width=True,
                )

    with tab3:
        st.subheader("Hashtag Explorer")
        st.caption("Pull top posts for any hashtag to see if it's worth using.")

        col_a, col_b = st.columns([3, 1])
        with col_a:
            tag = st.text_input("Enter a hashtag", placeholder="#fitness")
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            explore = st.button("Explore", type="primary")

        if tag and explore:
            tag_clean = tag.lstrip("#")
            with st.spinner(f"Pulling posts for #{tag_clean}..."):
                try:
                    posts = scrape_hashtag(tag_clean, COMPETITOR_MAX_POSTS)
                    if posts.empty:
                        st.error("No posts found.")
                    else:
                        avg_eng = posts["engagement"].mean()
                        st.metric(f"Avg engagement in #{tag_clean}", f"{avg_eng:.0f}")
                        top = posts.nlargest(10, "engagement")[
                            ["content_type", "likes", "comments", "engagement", "caption"]
                        ].reset_index(drop=True)
                        top["caption"] = top["caption"].str[:80] + "..."
                        st.dataframe(top, use_container_width=True)
                except Exception as e:
                    st.error(f"Explorer failed: {e}")
