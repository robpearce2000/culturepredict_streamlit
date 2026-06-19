import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Instagram Growth OS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh live feed on startup ─────────────────────────────────────────
# Runs once per session. If auto_refresh is on and cache is stale, silently
# re-scrapes the user's own profile before rendering any page.
if not st.session_state.get("_startup_checked"):
    st.session_state["_startup_checked"] = True
    try:
        from live_feed import load_live_settings, cache_is_stale, refresh_and_save
        live = load_live_settings()
        handle = live.get("handle", "")
        auto = live.get("auto_refresh", False)
        hours = live.get("refresh_hours", 24)
        apify_token = os.getenv("APIFY_API_TOKEN", "")

        if handle and auto and apify_token and cache_is_stale(hours):
            with st.spinner(f"Auto-refreshing data for @{handle}..."):
                posts, err = refresh_and_save(handle, apify_token)
                if not err:
                    st.toast(f"Data refreshed — {len(posts)} posts from @{handle}", icon="✅")
                else:
                    st.toast(f"Auto-refresh failed: {err}", icon="⚠️")

        # Restore live settings into session state
        if handle:
            st.session_state.setdefault("live_handle", handle)
            st.session_state.setdefault("auto_refresh", auto)
            st.session_state.setdefault("refresh_hours", hours)
    except Exception:
        pass  # Never block startup

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Instagram Growth OS")
    st.markdown("---")

    # Show data freshness indicator
    try:
        from live_feed import cache_age_hours, load_live_settings
        from cache import cache_age_str
        age_h = cache_age_hours()
        live_cfg = load_live_settings()
        threshold = live_cfg.get("refresh_hours", 24)
        if age_h < 9999:
            icon = "🟢" if age_h < threshold else "🟡"
            st.caption(f"{icon} Data: {cache_age_str()}")
    except Exception:
        pass

    page = st.radio("", [
        "🏠 Home",
        "📊 My Analytics",
        "📈 Growth Trends",
        "⭐ Post Scoring",
        "✍️ Caption Analyser",
        "🗓️ Content Calendar",
        "🔍 Competitor Spy",
        "⚔️ Head-to-Head",
        "🌐 Niche Leaderboard",
        "🎬 Video Formulas",
        "🪝 Hook Library",
        "💡 Content Ideas",
        "🎯 Viral Predictor",
        "🕐 Posting Schedule",
        "# Hashtag Strategy",
        "📥 Export Report",
        "⚙️ Settings",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Set up Live Feed in ⚙️ Settings to auto-refresh your data daily.")

# ── Route to pages — app.py is routing only, no logic here ───────────────────
if page == "🏠 Home":
    import pages.home as p; p.render()
elif page == "📊 My Analytics":
    import pages.my_analytics as p; p.render()
elif page == "📈 Growth Trends":
    import pages.growth_trends as p; p.render()
elif page == "⭐ Post Scoring":
    import pages.post_scoring as p; p.render()
elif page == "✍️ Caption Analyser":
    import pages.caption_analyzer_page as p; p.render()
elif page == "🗓️ Content Calendar":
    import pages.content_calendar as p; p.render()
elif page == "🔍 Competitor Spy":
    import pages.competitor_spy as p; p.render()
elif page == "⚔️ Head-to-Head":
    import pages.head_to_head as p; p.render()
elif page == "🌐 Niche Leaderboard":
    import pages.multi_competitor as p; p.render()
elif page == "🎬 Video Formulas":
    import pages.video_formulas as p; p.render()
elif page == "🪝 Hook Library":
    import pages.hook_library_page as p; p.render()
elif page == "💡 Content Ideas":
    import pages.content_ideas_page as p; p.render()
elif page == "🎯 Viral Predictor":
    import pages.viral_predictor as p; p.render()
elif page == "🕐 Posting Schedule":
    import pages.posting_schedule as p; p.render()
elif page == "# Hashtag Strategy":
    import pages.hashtag_strategy as p; p.render()
elif page == "📥 Export Report":
    import pages.export_report as p; p.render()
elif page == "⚙️ Settings":
    import pages.settings as p; p.render()
