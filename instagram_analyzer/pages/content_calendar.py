"""
Content calendar — visual monthly grid with posts coloured by performance score.
Shows posting gaps, consistency streaks, and which days perform best.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import calendar
from datetime import date

from cache import load_analysis
from scoring import score_posts


_SCORE_COLORS = {
    "🔥 Viral": "#ff6b6b",
    "✅ Strong": "#E1306C",
    "👍 Average": "#833AB4",
    "📉 Weak": "#555",
    "❌ Flopped": "#333",
}


def _calendar_heatmap(posts: pd.DataFrame) -> go.Figure:
    """GitHub-style contribution heatmap: x=week, y=day-of-week."""
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["engagement"] = df["likes"] + df["comments"]

    daily = df.groupby("date").agg(
        engagement=("engagement", "sum"),
        posts=("engagement", "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["weekday"] = daily["date"].dt.weekday          # 0=Mon
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["year"] = daily["date"].dt.year

    # Build grid
    fig = go.Figure(go.Scatter(
        x=daily["date"],
        y=daily["weekday"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}),
        mode="markers",
        marker=dict(
            size=daily["posts"].clip(upper=5) * 6 + 6,
            color=daily["engagement"],
            colorscale="RdPu",
            showscale=True,
            colorbar=dict(title="Engagement"),
            line=dict(width=0),
        ),
        text=daily.apply(
            lambda r: f"{r['date'].strftime('%b %d')}<br>{r['posts']} post(s)<br>{r['engagement']:.0f} engagement",
            axis=1
        ),
        hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Posting Activity Heatmap",
        xaxis_title="Date",
        yaxis=dict(
            categoryorder="array",
            categoryarray=["Sun","Sat","Fri","Thu","Wed","Tue","Mon"],
        ),
        height=300,
    )
    return fig


def _monthly_grid(posts: pd.DataFrame, year: int, month: int) -> go.Figure:
    """Calendar grid for a single month, cells coloured by post score."""
    df = posts.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"]
    scored = score_posts(df)
    scored["date"] = scored["timestamp"].dt.date

    # Filter to selected month
    mask = (scored["timestamp"].dt.year == year) & (scored["timestamp"].dt.month == month)
    month_posts = scored[mask].copy()

    cal = calendar.monthcalendar(year, month)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Build annotation grid
    z = [[0]*7 for _ in cal]
    text = [["" for _ in range(7)] for _ in cal]
    hover = [["" for _ in range(7)] for _ in cal]

    for week_idx, week in enumerate(cal):
        for day_idx, day_num in enumerate(week):
            if day_num == 0:
                text[week_idx][day_idx] = ""
                continue
            d = date(year, month, day_num)
            day_posts = month_posts[month_posts["date"] == d]
            text[week_idx][day_idx] = str(day_num)
            if not day_posts.empty:
                best_score = day_posts["score"].max()
                z[week_idx][day_idx] = best_score
                verdicts = ", ".join(day_posts["verdict"].tolist())
                total_eng = int(day_posts["engagement"].sum())
                hover[week_idx][day_idx] = (
                    f"<b>{d.strftime('%b %d')}</b><br>"
                    f"{len(day_posts)} post(s)<br>"
                    f"Best score: {best_score:.1f}<br>"
                    f"Total engagement: {total_eng}<br>"
                    f"{verdicts}"
                )
            else:
                z[week_idx][day_idx] = -1
                hover[week_idx][day_idx] = f"<b>{d.strftime('%b %d')}</b><br>No posts"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=days,
        colorscale=[
            [0, "#1a1a2e"], [0.1, "#222"], [0.3, "#833AB4"],
            [0.6, "#E1306C"], [1.0, "#ff6b6b"],
        ],
        zmin=-1, zmax=10,
        text=text,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        showscale=False,
    ))
    # Overlay day numbers
    for week_idx, week in enumerate(cal):
        for day_idx, day_num in enumerate(week):
            if day_num > 0:
                fig.add_annotation(
                    x=days[day_idx], y=week_idx,
                    text=str(day_num), showarrow=False,
                    font=dict(color="white", size=12),
                )

    fig.update_layout(
        template="plotly_dark",
        title=f"{calendar.month_name[month]} {year}",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed", showticklabels=False),
        height=280,
    )
    return fig


def render():
    st.title("📅 Content Calendar")
    st.caption("Visual view of when you post and how each post performed — spot gaps and streaks instantly.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return

    posts = cached["posts"]
    if posts.empty:
        st.warning("No post data found.")
        return

    posts = posts.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")
    posts = posts.dropna(subset=["timestamp"])
    posts["likes"] = posts["likes"].fillna(0)
    posts["comments"] = posts["comments"].fillna(0)

    if posts.empty:
        st.warning("No valid timestamps found in your post data.")
        return

    # ── Activity heatmap (full history) ──────────────────────────────────────
    st.plotly_chart(_calendar_heatmap(posts), use_container_width=True)

    st.markdown("---")

    # ── Monthly grid picker ───────────────────────────────────────────────────
    st.subheader("Monthly View")

    available_months = (
        posts["timestamp"].dt.to_period("M")
        .drop_duplicates()
        .sort_values(ascending=False)
        .astype(str)
        .tolist()
    )

    selected = st.selectbox("Select month", available_months)
    if selected:
        year, month = int(selected[:4]), int(selected[5:7])
        st.plotly_chart(_monthly_grid(posts, year, month), use_container_width=True)

        # Summary for selected month
        mask = (posts["timestamp"].dt.year == year) & (posts["timestamp"].dt.month == month)
        month_posts = posts[mask].copy()
        month_posts["engagement"] = month_posts["likes"] + month_posts["comments"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Posts This Month", len(month_posts))
        if not month_posts.empty:
            c2.metric("Avg Engagement", f"{month_posts['engagement'].mean():.0f}")
            c3.metric("Best Day", month_posts.loc[month_posts["engagement"].idxmax(), "timestamp"].strftime("%A %d"))

    st.markdown("---")

    # ── Posting consistency stats ─────────────────────────────────────────────
    st.subheader("Posting Consistency")
    posts_sorted = posts.sort_values("timestamp")
    posts_sorted["gap_days"] = posts_sorted["timestamp"].diff().dt.days

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Days Between Posts", f"{posts_sorted['gap_days'].mean():.1f}")
    c2.metric("Longest Gap (days)", f"{posts_sorted['gap_days'].max():.0f}")
    days_active = (posts_sorted["timestamp"].max() - posts_sorted["timestamp"].min()).days + 1
    posts_per_week = len(posts) / max(days_active / 7, 1)
    c3.metric("Posts / Week (avg)", f"{posts_per_week:.1f}")

    # Best day of week
    posts_sorted["dayofweek"] = posts_sorted["timestamp"].dt.day_name()
    posts_sorted["engagement"] = posts_sorted["likes"] + posts_sorted["comments"]
    day_perf = posts_sorted.groupby("dayofweek")["engagement"].mean().sort_values(ascending=False)
    if not day_perf.empty:
        st.markdown(f"**Best day to post:** {day_perf.index[0]} (avg {day_perf.iloc[0]:.0f} engagement)")
