import streamlit as st
import anthropic

from cache import load_analysis
from trends import monthly_stats, trend_chart, content_type_trend, growth_velocity, best_and_worst_months
from prompts import trend_interpretation
from config import CLAUDE_MODEL, AI_MAX_TOKENS_SUMMARY


def render():
    st.title("📈 Growth Trends")
    st.caption("Month-on-month breakdown — see if your account is accelerating, plateauing, or declining.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return

    posts = cached["posts"]
    if posts.empty:
        st.warning("No post data found.")
        return

    monthly = monthly_stats(posts)
    if len(monthly) < 2:
        st.warning("Need at least 2 months of posts to show trends. Upload a longer export.")
        return

    vel = growth_velocity(monthly)
    best, worst = best_and_worst_months(monthly)

    # ── Trend summary metrics ─────────────────────────────────────────────────
    latest = vel.iloc[-1]
    prev = vel.iloc[-2]
    change = latest["avg_engagement"] - prev["avg_engagement"]
    change_pct = (change / prev["avg_engagement"] * 100) if prev["avg_engagement"] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("This Month Avg Engagement", f"{latest['avg_engagement']:.0f}",
              delta=f"{change_pct:+.1f}% vs last month",
              delta_color="normal")
    c2.metric("Posts This Month", int(latest["posts"]))
    c3.metric("Best Month", str(best.get("month", "N/A")),
              delta=f"{best.get('avg_engagement', 0):.0f} avg eng")
    c4.metric("Worst Month", str(worst.get("month", "N/A")),
              delta=f"{worst.get('avg_engagement', 0):.0f} avg eng", delta_color="off")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(trend_chart(posts), use_container_width=True)
    st.plotly_chart(content_type_trend(posts), use_container_width=True)

    # ── Monthly table ─────────────────────────────────────────────────────────
    with st.expander("Monthly breakdown table"):
        display = vel.copy()
        display["month"] = display["month"].astype(str)
        display.columns = ["Month", "Posts", "Avg Engagement", "Total Likes",
                           "Total Comments", "Engagement Change %", "Posts Change"]
        st.dataframe(display, use_container_width=True)

    st.markdown("---")

    # ── AI trend interpretation ───────────────────────────────────────────────
    st.subheader("AI Trend Analysis")
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings.")
    else:
        if st.button("Interpret My Trends", type="primary"):
            with st.spinner("Analysing your growth trajectory..."):
                try:
                    monthly_text = vel[["month", "posts", "avg_engagement", "engagement_change_pct"]].to_string(index=False)
                    client = anthropic.Anthropic(api_key=api_key)
                    msg = client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=AI_MAX_TOKENS_SUMMARY,
                        messages=[{"role": "user", "content": trend_interpretation(
                            monthly_text,
                            f"{best.get('month')} ({best.get('avg_engagement', 0):.0f} avg engagement)",
                            f"{worst.get('month')} ({worst.get('avg_engagement', 0):.0f} avg engagement)",
                        )}],
                    )
                    st.session_state["trend_analysis"] = msg.content[0].text
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "trend_analysis" in st.session_state:
            st.markdown(st.session_state["trend_analysis"])
