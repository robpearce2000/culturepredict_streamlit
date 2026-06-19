import streamlit as st
import pandas as pd

from cache import load_analysis, list_cached_competitors, load_competitor
from content_ideas import generate_ideas, generate_caption
from config import CONTENT_IDEAS_COUNT


def render():
    st.title("💡 Content Ideas")
    st.caption("AI generates post ideas from what's already working in your account and your competitors' — not generic advice.")

    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings.")
        return

    my_data = load_analysis()
    my_posts = my_data["posts"] if my_data else pd.DataFrame()

    competitors = list_cached_competitors()
    comp_posts = pd.DataFrame()
    if competitors:
        selected_comp = st.selectbox("Include competitor data from", ["None"] + competitors)
        if selected_comp != "None":
            comp_data = load_competitor(selected_comp.lstrip("@"))
            if comp_data:
                comp_posts = comp_data["posts"]

    niche = st.text_input("Your niche (optional — makes ideas more specific)", placeholder="e.g. fitness for busy parents, personal finance UK")
    n_ideas = st.slider("Number of ideas", min_value=5, max_value=20, value=CONTENT_IDEAS_COUNT)

    st.markdown("---")

    if st.button("Generate Ideas", type="primary"):
        if my_posts.empty and comp_posts.empty:
            st.warning("Upload your Instagram export or analyse a competitor first to get data-driven ideas.")
        else:
            with st.spinner("Generating ideas based on your data..."):
                try:
                    ideas = generate_ideas(my_posts, comp_posts, api_key, niche, n_ideas)
                    st.session_state["content_ideas"] = ideas
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if "content_ideas" in st.session_state:
        st.markdown(st.session_state["content_ideas"])

        st.markdown("---")
        st.subheader("Caption Writer")
        st.caption("Paste any idea above and get a ready-to-post caption.")
        idea_input = st.text_area("Paste an idea here", height=80)
        style = st.selectbox("Caption style", ["conversational", "bold and direct", "storytelling", "educational"])
        if st.button("Write Caption") and idea_input:
            with st.spinner("Writing caption..."):
                try:
                    caption = generate_caption(idea_input, api_key, style)
                    st.session_state["generated_caption"] = caption
                except Exception as e:
                    st.error(f"Caption generation failed: {e}")

        if "generated_caption" in st.session_state:
            st.text_area("Your caption", value=st.session_state["generated_caption"], height=200)
            st.caption("Copy, edit, post. Done.")
