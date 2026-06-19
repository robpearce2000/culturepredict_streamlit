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

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Instagram Growth OS")
    st.markdown("---")
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
    st.caption("Upload your Instagram export ZIP in **My Analytics** to unlock all features.")

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
