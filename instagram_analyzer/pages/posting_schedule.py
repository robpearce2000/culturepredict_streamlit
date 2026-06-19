import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from cache import load_analysis, list_cached_competitors, load_competitor
from schedule import best_slots, generate_weekly_schedule, DAYS


def _slots_chart(slots: list[dict]) -> go.Figure:
    if not slots:
        return go.Figure().update_layout(title="No timing data", template="plotly_dark")

    fig = go.Figure()
    for slot in slots:
        day_idx = DAYS.index(slot["day"]) if slot["day"] in DAYS else 0
        fig.add_trace(go.Scatter(
            x=[slot["time"]],
            y=[slot["day"]],
            mode="markers+text",
            marker=dict(size=slot["avg_engagement"] / 50 + 15, color="#E1306C", opacity=0.8),
            text=[f"  {slot['avg_engagement']:.0f}"],
            textposition="middle right",
            name=f"{slot['day']} {slot['time']}",
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Your Best Posting Slots (bubble size = avg engagement)",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(DAYS))),
        xaxis_title="Time of Day",
        height=350,
    )
    return fig


def render():
    st.title("📅 Posting Schedule")
    st.caption("Your optimal weekly content calendar — built from your actual performance data, not generic advice.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** to generate a personalised schedule.")
        return

    posts = cached["posts"]
    competitors = list_cached_competitors()
    comp_posts = pd.DataFrame()

    # ── Settings ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        posts_per_week = st.slider("Posts per week", min_value=3, max_value=14, value=5)
    with col2:
        niche = st.text_input("Your niche", placeholder="e.g. UK fitness")
    with col3:
        if competitors:
            selected_comp = st.selectbox("Include competitor data", ["None"] + competitors)
            if selected_comp != "None":
                comp_data = load_competitor(selected_comp.lstrip("@"))
                if comp_data:
                    comp_posts = comp_data["posts"]

    st.markdown("---")

    # ── Best slots visualisation ───────────────────────────────────────────────
    slots = best_slots(posts, n_slots=posts_per_week)
    if slots:
        st.subheader("Your Best Posting Times")
        st.plotly_chart(_slots_chart(slots), use_container_width=True)

        cols = st.columns(len(slots))
        for col, slot in zip(cols, slots):
            col.metric(slot["day"], slot["time"], delta=f"{slot['avg_engagement']:.0f} avg eng")
    else:
        st.info("Not enough timing data yet — the schedule will use proven default times.")

    st.markdown("---")

    # ── AI schedule ───────────────────────────────────────────────────────────
    st.subheader("AI-Generated Weekly Schedule")
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings.")
    else:
        if st.button("Generate This Week's Schedule", type="primary"):
            with st.spinner("Building your schedule..."):
                try:
                    schedule = generate_weekly_schedule(
                        posts, comp_posts, api_key, niche, posts_per_week
                    )
                    st.session_state["weekly_schedule"] = schedule
                except Exception as e:
                    st.error(f"Schedule generation failed: {e}")

        if "weekly_schedule" in st.session_state:
            st.markdown(
                f"""<div style="background:#0e0e1a;border:1px solid #333;padding:20px 24px;border-radius:8px;line-height:1.8">
{st.session_state["weekly_schedule"]}
</div>""",
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download Schedule",
                data=st.session_state["weekly_schedule"],
                file_name="instagram_schedule.txt",
                mime="text/plain",
            )
