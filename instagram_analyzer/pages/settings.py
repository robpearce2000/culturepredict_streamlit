import streamlit as st
import os
from pathlib import Path

from config import CACHE_DIR, APIFY_PROFILE_ACTOR, APIFY_HASHTAG_ACTOR, CLAUDE_MODEL
from cache import list_cached_competitors, cache_age_str
from live_feed import load_live_settings, save_live_settings, cache_age_hours, cache_is_stale


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
        help="Get yours at apify.com — needed for live data feed and competitor scraping",
    )

    follower_count = st.number_input(
        "Follower Count (for engagement rate %)",
        min_value=0,
        value=st.session_state.get("follower_count", 0),
        step=100,
        help="Used in Post Scoring to calculate engagement rate %. Not stored persistently.",
    )

    if st.button("Save Settings", type="primary"):
        st.session_state["api_key"] = anthropic_key
        st.session_state["apify_key"] = apify_key
        st.session_state["follower_count"] = int(follower_count)
        os.environ["APIFY_API_TOKEN"] = apify_key
        st.success("Settings saved for this session.")

    st.markdown("---")

    # ── Live Data Feed ────────────────────────────────────────────────────────
    st.subheader("Live Data Feed")
    st.caption("Scrape your own profile via Apify so the app always shows your latest posts — no ZIP export needed.")

    live = load_live_settings()

    my_handle = st.text_input(
        "Your Instagram handle",
        value=live.get("handle", ""),
        placeholder="e.g. cristiano (no @)",
    )
    refresh_hours = st.selectbox(
        "Auto-refresh every",
        options=[6, 12, 24, 48, 72],
        index=[6, 12, 24, 48, 72].index(live.get("refresh_hours", 24)),
        format_func=lambda h: f"{h} hours",
    )
    auto_refresh = st.toggle(
        "Auto-refresh when app opens (if data is older than above)",
        value=live.get("auto_refresh", False),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Live Feed Settings"):
            save_live_settings(my_handle, refresh_hours, auto_refresh)
            st.session_state["live_handle"] = my_handle
            st.session_state["auto_refresh"] = auto_refresh
            st.session_state["refresh_hours"] = refresh_hours
            st.success("Live feed settings saved.")

    with col2:
        apify_ready = st.session_state.get("apify_key", os.getenv("APIFY_API_TOKEN", ""))
        if st.button("Refresh My Data Now", disabled=not (my_handle and apify_ready)):
            if not my_handle:
                st.error("Enter your handle first.")
            elif not apify_ready:
                st.error("Add your Apify API Token above and save settings first.")
            else:
                with st.spinner(f"Scraping @{my_handle} from Instagram..."):
                    from live_feed import refresh_and_save
                    posts, err = refresh_and_save(my_handle, apify_ready)
                    if err:
                        st.error(f"Scrape failed: {err}")
                    else:
                        st.success(f"Done — {len(posts)} posts loaded from @{my_handle}.")
                        st.rerun()

    # Show current data status
    age_h = cache_age_hours()
    if age_h < 9999:
        stale = cache_is_stale(refresh_hours)
        status = f"⚠️ Stale ({age_h:.0f}h old)" if stale else f"✅ Fresh ({age_h:.1f}h old)"
        st.caption(f"Current data: {status}")
    else:
        st.caption("No data cached yet.")

    if not apify_ready:
        st.info("Add your Apify token above to enable live scraping. Without it the app uses your uploaded ZIP export.")

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
        st.info("No account data cached yet. Upload your export in My Analytics or use Live Feed above.")

    competitors = list_cached_competitors()
    if competitors:
        st.write(f"Competitors cached: {', '.join('@' + h for h in competitors)}")
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
