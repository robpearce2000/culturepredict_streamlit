import streamlit as st
import anthropic

from cache import load_analysis
from caption_analyzer import (
    top_words_by_engagement, top_phrases_by_engagement,
    caption_length_buckets, caption_feature_impact,
    word_lift_chart, caption_length_chart, feature_impact_chart,
)
from config import CLAUDE_MODEL, AI_MAX_TOKENS_SUMMARY


def render():
    st.title("✍️ Caption Analyser")
    st.caption("Find the words, phrases, and structures that drive engagement in your captions.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return

    posts = cached["posts"]
    if posts.empty or "caption" not in posts.columns:
        st.warning("No caption data found.")
        return

    posts = posts.copy()
    posts["likes"] = posts["likes"].fillna(0)
    posts["comments"] = posts["comments"].fillna(0)

    tab1, tab2, tab3, tab4 = st.tabs(["Word Power", "Caption Length", "Feature Impact", "AI Verdict"])

    with tab1:
        st.subheader("Words That Drive Engagement")
        st.caption("Lift > 1.0 means this word appears more in high-engagement captions than low ones.")
        word_df = top_words_by_engagement(posts)
        if word_df.empty:
            st.info("Not enough caption data to analyse words yet.")
        else:
            st.plotly_chart(word_lift_chart(word_df), use_container_width=True)
            with st.expander("Full word table"):
                st.dataframe(word_df.rename(columns={
                    "word": "Word", "lift": "Lift Score",
                    "high_count": "Appearances in High-Eng Posts",
                })[["Word", "Lift Score", "Appearances in High-Eng Posts"]], use_container_width=True)

        st.subheader("Two-Word Phrases")
        phrase_df = top_phrases_by_engagement(posts)
        if not phrase_df.empty:
            st.dataframe(phrase_df.rename(columns={"phrase": "Phrase", "lift": "Lift", "count": "Count"}),
                         use_container_width=True)

    with tab2:
        st.subheader("Caption Length vs Engagement")
        bucket_df = caption_length_buckets(posts)
        st.plotly_chart(caption_length_chart(bucket_df), use_container_width=True)
        st.dataframe(bucket_df.rename(columns={
            "length_bucket": "Length Bucket",
            "avg_engagement": "Avg Engagement",
            "post_count": "Posts",
        }), use_container_width=True)

    with tab3:
        st.subheader("Caption Features: Do They Move the Needle?")
        feat_df = caption_feature_impact(posts)
        st.plotly_chart(feature_impact_chart(feat_df), use_container_width=True)
        st.dataframe(feat_df.rename(columns={
            "feature": "Feature", "with_avg": "Avg Eng WITH",
            "without_avg": "Avg Eng WITHOUT", "lift_pct": "Lift %",
            "posts_with": "Posts With Feature",
        }), use_container_width=True)

    with tab4:
        st.subheader("AI Caption Verdict")
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            st.warning("Add your Claude API key in ⚙️ Settings.")
        else:
            if st.button("Analyse My Caption Strategy", type="primary"):
                with st.spinner("Analysing your captions..."):
                    try:
                        word_df = top_words_by_engagement(posts)
                        bucket_df = caption_length_buckets(posts)
                        feat_df = caption_feature_impact(posts)

                        context = f"""Caption analysis data:

Top words in high-engagement captions (lift score):
{word_df[["word","lift","high_count"]].head(10).to_string(index=False) if not word_df.empty else "Insufficient data"}

Performance by caption length:
{bucket_df.to_string(index=False)}

Feature impact:
{feat_df[["feature","with_avg","without_avg","lift_pct"]].to_string(index=False)}"""

                        client = anthropic.Anthropic(api_key=api_key)
                        msg = client.messages.create(
                            model=CLAUDE_MODEL,
                            max_tokens=AI_MAX_TOKENS_SUMMARY,
                            messages=[{"role": "user", "content": f"""You are an Instagram copywriting analyst.

Rules:
- Reference specific numbers from the data
- Give 4 concrete caption rules for this creator specifically
- Prescriptive: say exactly what to write, not what to "consider"

{context}

Output 4 bullet points: one on word choice, one on length, one on features (questions/CTAs/emojis), one on the single highest-leverage change to make immediately."""}],
                        )
                        st.session_state["caption_analysis"] = msg.content[0].text
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            if "caption_analysis" in st.session_state:
                st.markdown(st.session_state["caption_analysis"])
