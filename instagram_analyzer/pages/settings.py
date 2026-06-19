import streamlit as st
import os
from pathlib import Path

from config import CACHE_DIR, APIFY_PROFILE_ACTOR, APIFY_HASHTAG_ACTOR, CLAUDE_MODEL
from cache import list_cached_competitors, cache_age_str


def render():
    st.title("⚙️ Settings")

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.subheader("API Keys")
    st.caption("Keys are stored in session only — never saved to disk or committed to git.")

    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.get("api_key", os.getenv("ANTHROPIC_API_KEY", "")),
        help="Get yours at console.anthropic.com",
    )
    apify_key = st.text_input(
        "Apify API Token",
        type="password",
        value=st.session_state.get("apify_key", os.getenv("APIFY_API_TOKEN", "")),
        help="Get yours at apify.com — needed for live competitor scraping",
    )

    if st.button("Save Keys", type="primary"):
        st.session_state["api_key"] = anthropic_key
        st.session_state["apify_key"] = apify_key
        os.environ["APIFY_API_TOKEN"] = apify_key
        st.success("Keys saved for this session.")

    st.markdown("---")

    # ── Cache status ──────────────────────────────────────────────────────────
    st.subheader("Cached Data")
    st.caption(f"Stored in: `{CACHE_DIR}`")

    my_cache = CACHE_DIR / "my_account.json"
    if my_cache.exists():
        st.success(f"My account data: {cache_age_str()}")
        if st.button("Clear my account cache"):
            my_cache.unlink()
            st.rerun()
    else:
        st.info("No account data cached yet. Upload your export in My Analytics.")

    competitors = list_cached_competitors()
    if competitors:
        st.write(f"Competitors cached: {', '.join(competitors)}")
        if st.button("Clear all competitor caches"):
            for f in CACHE_DIR.glob("competitor_*.json"):
                f.unlink()
            st.rerun()

    st.markdown("---")

    # ── Config info ───────────────────────────────────────────────────────────
    st.subheader("Configuration")
    st.code(f"""AI Model:              {CLAUDE_MODEL}
Profile scraper actor: {APIFY_PROFILE_ACTOR}
Hashtag scraper actor: {APIFY_HASHTAG_ACTOR}
Cache directory:       {CACHE_DIR}
""")

    st.markdown("---")

    # ── .env reminder ─────────────────────────────────────────────────────────
    st.subheader("Local Setup")
    st.markdown("""
Create a `.env` file in the `instagram_analyzer/` folder (never commit this):
```
ANTHROPIC_API_KEY=your_key_here
APIFY_API_TOKEN=your_token_here
```
See `.env.example` for the template.
    """)
