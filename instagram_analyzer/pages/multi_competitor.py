"""
Multi-competitor league table — all cached competitors ranked side-by-side.
Shows who dominates the niche and which metrics to target.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from cache import load_analysis, list_cached_competitors, load_competitor


def _build_league_table(cached_handles: list[str]) -> pd.DataFrame:
    rows = []
    for handle in cached_handles:
        data = load_competitor(handle)
        if data is None:
            continue
        comp = data["posts"].copy()
        if comp.empty:
            continue
        comp["engagement"] = comp["likes"] + comp["comments"]
        comp["timestamp"] = pd.to_datetime(comp["timestamp"], errors="coerce")

        row = {
            "Handle": f"@{handle}",
            "Posts Analysed": len(comp),
            "Avg Engagement": round(comp["engagement"].mean(), 0),
            "Avg Likes": round(comp["likes"].mean(), 0),
            "Avg Comments": round(comp["comments"].mean(), 0),
            "Top Post Eng": int(comp["engagement"].max()),
            "Best Content": comp.groupby("content_type")["engagement"].mean().idxmax() if "content_type" in comp else "N/A",
            "Posting Freq / wk": _posting_freq(comp),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Avg Engagement", ascending=False).reset_index(drop=True)


def _posting_freq(posts: pd.DataFrame) -> float:
    posts = posts.dropna(subset=["timestamp"])
    if len(posts) < 2:
        return 0.0
    span = (posts["timestamp"].max() - posts["timestamp"].min()).days
    return round(len(posts) / max(span / 7, 1), 1)


def _radar_chart(league: pd.DataFrame) -> go.Figure:
    metrics = ["Avg Engagement", "Avg Likes", "Avg Comments"]
    available = [m for m in metrics if m in league.columns]
    if not available or league.empty:
        return go.Figure()

    fig = go.Figure()
    for _, row in league.iterrows():
        # Normalise 0-100 within dataset
        vals = [row[m] for m in available]
        maxes = [league[m].max() for m in available]
        norm = [round(v / max(mx, 1) * 100, 1) for v, mx in zip(vals, maxes)]
        fig.add_trace(go.Scatterpolar(
            r=norm + [norm[0]],
            theta=available + [available[0]],
            fill="toself",
            name=row["Handle"],
            opacity=0.7,
        ))
    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Competitor Radar (normalised to top performer)",
        height=420,
    )
    return fig


def _engagement_bar(league: pd.DataFrame) -> go.Figure:
    colors = ["#E1306C", "#833AB4", "#F77737", "#FCAF45", "#FFDC80"]
    fig = go.Figure(go.Bar(
        x=league["Handle"],
        y=league["Avg Engagement"],
        marker_color=colors[:len(league)],
        text=league["Avg Engagement"].astype(int),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Average Engagement — Niche Leaderboard",
        yaxis_title="Avg Engagement per Post",
        height=380,
    )
    return fig


def render():
    st.title("🏆 Niche Leaderboard")
    st.caption("All cached competitors ranked — see who dominates your niche and where the gaps are.")

    handles = list_cached_competitors()

    # Also include own account as "You" if available
    my_data = load_analysis()
    my_row = None
    if my_data and not my_data["posts"].empty:
        my_posts = my_data["posts"].copy()
        my_posts["engagement"] = my_posts["likes"] + my_posts["comments"]
        my_posts["timestamp"] = pd.to_datetime(my_posts["timestamp"], errors="coerce")
        profile = my_data.get("profile", {})
        handle_label = profile.get("username", "you") if profile else "you"
        my_row = {
            "Handle": f"⭐ @{handle_label} (you)",
            "Posts Analysed": len(my_posts),
            "Avg Engagement": round(my_posts["engagement"].mean(), 0),
            "Avg Likes": round(my_posts["likes"].mean(), 0),
            "Avg Comments": round(my_posts["comments"].mean(), 0),
            "Top Post Eng": int(my_posts["engagement"].max()),
            "Best Content": my_posts.groupby("content_type")["engagement"].mean().idxmax() if "content_type" in my_posts else "N/A",
            "Posting Freq / wk": _posting_freq(my_posts),
        }

    if not handles and my_row is None:
        st.info("No competitor data yet. Use **Competitor Spy** to scrape some profiles, then come back.")
        return

    league = _build_league_table(handles)
    if my_row:
        league = pd.concat([pd.DataFrame([my_row]), league], ignore_index=True)

    if league.empty:
        st.warning("Could not build leaderboard — no valid data in cache.")
        return

    # ── Leaderboard table ──────────────────────────────────────────────────────
    st.subheader(f"Leaderboard — {len(league)} accounts")
    st.dataframe(league, use_container_width=True)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    if len(league) >= 2:
        st.plotly_chart(_engagement_bar(league), use_container_width=True)
        st.plotly_chart(_radar_chart(league), use_container_width=True)
    else:
        st.info("Scrape at least one more competitor to unlock comparison charts.")

    st.markdown("---")

    # ── Gap analysis ───────────────────────────────────────────────────────────
    st.subheader("Gap Analysis")
    if my_row and len(league) >= 2:
        comps = league[~league["Handle"].str.startswith("⭐")]
        if not comps.empty:
            best_comp_eng = comps["Avg Engagement"].max()
            my_eng = my_row["Avg Engagement"]
            gap = best_comp_eng - my_eng
            gap_pct = gap / max(my_eng, 1) * 100
            leader = comps.loc[comps["Avg Engagement"].idxmax(), "Handle"]
            if gap > 0:
                st.metric("Gap to niche leader", f"{gap:.0f} engagement",
                          delta=f"{gap_pct:.0f}% behind {leader}", delta_color="inverse")
            else:
                st.metric("You vs niche leader", "You're ahead!",
                          delta=f"{abs(gap):.0f} engagement lead", delta_color="normal")
    elif not handles:
        st.info("Add competitors via **Competitor Spy** to see your gap.")

    st.markdown("---")
    st.subheader("Manage Competitors")
    if handles:
        st.caption(f"{len(handles)} competitor(s) in cache: {', '.join('@' + h for h in handles)}")
        st.info("To add more, go to **Competitor Spy**. To remove, clear the cache in ⚙️ Settings.")
    else:
        st.info("No competitors cached yet.")
