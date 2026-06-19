import streamlit as st

from cache import list_cached_competitors, load_competitor
from video_structure import extract_video_formula, analyse_content_mix


def render():
    st.title("🎬 Video Formulas")
    st.caption("AI reverse-engineers the exact Reel structure your competitors use — so you can copy the formula, not the content.")

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings to use this feature.")
        return

    competitors = list_cached_competitors()
    if not competitors:
        st.info("No competitors tracked yet. Go to **Competitor Spy** and analyse a @handle first.")
        return

    handle = st.selectbox("Choose a competitor", competitors).lstrip("@")
    data = load_competitor(handle)
    if not data:
        st.error("Could not load competitor data.")
        return

    posts = data["posts"]
    posts["engagement"] = posts["likes"] + posts["comments"]

    reels = posts[posts["content_type"].isin(["Reel", "Video"])]
    st.metric("Reels/Videos found", len(reels))

    st.markdown("---")
    tab1, tab2 = st.tabs(["Reel Formula", "Content Mix Analysis"])

    with tab1:
        st.subheader(f"@{handle}'s Reel Formula")
        if reels.empty:
            st.warning("No Reels or video content found for this account.")
        else:
            if st.button("Extract Formula", type="primary"):
                with st.spinner("Analysing their top Reels..."):
                    try:
                        formula = extract_video_formula(posts, handle, api_key)
                        st.session_state[f"formula_{handle}"] = formula
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            key = f"formula_{handle}"
            if key in st.session_state:
                st.markdown(st.session_state[key])

    with tab2:
        st.subheader(f"@{handle}'s Content Strategy")
        if st.button("Analyse Content Mix", type="primary"):
            with st.spinner("Analysing their content mix..."):
                try:
                    analysis = analyse_content_mix(posts, handle, api_key)
                    st.session_state[f"mix_{handle}"] = analysis
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        key = f"mix_{handle}"
        if key in st.session_state:
            st.markdown(st.session_state[key])
