import streamlit as st
import anthropic
import pandas as pd

from cache import load_analysis, list_cached_competitors, load_competitor
from benchmark import head_to_head_metrics, head_to_head_chart, posting_frequency_comparison, engagement_distribution
from prompts import head_to_head_summary
from config import CLAUDE_MODEL, AI_MAX_TOKENS_SUMMARY


def render():
    st.title("⚔️ Head-to-Head")
    st.caption("Your account benchmarked directly against a competitor — every metric side by side.")

    my_data = load_analysis()
    competitors = list_cached_competitors()

    if not my_data:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return
    if not competitors:
        st.info("Analyse a competitor in **Competitor Spy** first.")
        return

    my_posts = my_data["posts"]
    my_posts["engagement"] = my_posts["likes"] + my_posts["comments"]

    handle = st.selectbox("Compare against", competitors).lstrip("@")
    comp_data = load_competitor(handle)
    if not comp_data:
        st.error("Could not load competitor data.")
        return

    comp_posts = comp_data["posts"]
    comp_posts["engagement"] = comp_posts["likes"] + comp_posts["comments"]

    # ── Metric table ──────────────────────────────────────────────────────────
    metrics = head_to_head_metrics(my_posts, comp_posts, handle)

    st.subheader("At a Glance")
    # Highlight the winner in each row
    def highlight_winner(row):
        try:
            you_val = float(str(row["You"]).replace(",", ""))
            comp_val = float(str(row[f"@{handle}"]).replace(",", ""))
            if you_val > comp_val:
                return ["background-color: #2d1b2e", "background-color: #1a1a1a"]
            elif comp_val > you_val:
                return ["background-color: #1a1a1a", "background-color: #2d1b2e"]
        except Exception:
            pass
        return ["", ""]

    display = metrics.set_index("Account").T.reset_index()
    display.columns = ["Metric", "You", f"@{handle}"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(head_to_head_chart(my_posts, comp_posts, handle), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(engagement_distribution(my_posts, comp_posts, handle), use_container_width=True)
    with col2:
        st.plotly_chart(posting_frequency_comparison(my_posts, comp_posts, handle), use_container_width=True)

    st.markdown("---")

    # ── AI gap analysis ───────────────────────────────────────────────────────
    st.subheader("AI Gap Analysis")
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.warning("Add your Claude API key in ⚙️ Settings.")
    else:
        if st.button("Analyse the Gap", type="primary"):
            with st.spinner("Comparing your accounts..."):
                try:
                    my_metrics_text = metrics[metrics["Account"] == "You"].to_string(index=False)
                    comp_metrics_text = metrics[metrics["Account"] == f"@{handle}"].to_string(index=False)
                    client = anthropic.Anthropic(api_key=api_key)
                    msg = client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=AI_MAX_TOKENS_SUMMARY,
                        messages=[{"role": "user", "content": head_to_head_summary(
                            handle, my_metrics_text, comp_metrics_text
                        )}],
                    )
                    st.session_state[f"h2h_{handle}"] = msg.content[0].text
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        key = f"h2h_{handle}"
        if key in st.session_state:
            st.markdown(
                f"""<div style="background:#1a1a2e;border-left:4px solid #833AB4;padding:16px 20px;border-radius:6px">
{st.session_state[key]}
</div>""",
                unsafe_allow_html=True,
            )
