"""
Export report — download a full text or HTML insights report of the account analysis.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from cache import load_analysis, list_cached_competitors, load_competitor, cache_age_str
from scoring import score_posts, top_and_bottom
from trends import monthly_stats, growth_velocity, best_and_worst_months
from caption_analyzer import caption_length_buckets, caption_feature_impact


def _build_text_report(cached: dict) -> str:
    posts = cached["posts"].copy()
    followers = cached.get("followers", 0)
    profile = cached.get("profile", {}) or {}
    ai_summary = cached.get("ai_summary", "")

    posts["engagement"] = posts["likes"] + posts["comments"]
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")

    lines = []
    lines.append("=" * 60)
    lines.append("INSTAGRAM GROWTH OS — INSIGHTS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    lines.append(f"Data age: {cache_age_str()}")
    lines.append("=" * 60)

    # Profile
    username = profile.get("username", "Unknown")
    lines.append(f"\n👤 ACCOUNT: @{username}")
    if followers:
        lines.append(f"Followers: {followers:,}")
    lines.append(f"Posts analysed: {len(posts)}")
    if not posts.empty:
        lines.append(f"Date range: {posts['timestamp'].min().strftime('%b %Y')} – {posts['timestamp'].max().strftime('%b %Y')}")

    # Key metrics
    lines.append("\n📊 KEY METRICS")
    lines.append("-" * 40)
    lines.append(f"Avg engagement per post: {posts['engagement'].mean():.0f}")
    lines.append(f"Total likes: {posts['likes'].sum():,}")
    lines.append(f"Total comments: {posts['comments'].sum():,}")

    # Best content type
    if "content_type" in posts.columns:
        type_perf = posts.groupby("content_type")["engagement"].mean().sort_values(ascending=False)
        lines.append(f"\nBest content type: {type_perf.index[0]} ({type_perf.iloc[0]:.0f} avg engagement)")
        lines.append("Content type breakdown:")
        for ctype, eng in type_perf.items():
            lines.append(f"  {ctype}: {eng:.0f} avg engagement")

    # Top and bottom posts
    scored = score_posts(posts)
    top, bottom = top_and_bottom(scored, n=5)
    lines.append("\n🏆 TOP 5 POSTS")
    lines.append("-" * 40)
    for _, row in top.iterrows():
        lines.append(f"{row['verdict']} Score {row['score']:.1f} | {row.get('content_type','')} | {row['likes']:.0f}L {row['comments']:.0f}C")
        lines.append(f"  Caption: {row.get('caption','')[:80]}...")

    lines.append("\n❌ BOTTOM 5 POSTS")
    lines.append("-" * 40)
    for _, row in bottom.iterrows():
        lines.append(f"{row['verdict']} Score {row['score']:.1f} | {row.get('content_type','')} | {row['likes']:.0f}L {row['comments']:.0f}C")
        lines.append(f"  Caption: {row.get('caption','')[:80]}...")

    # Growth trends
    try:
        monthly = monthly_stats(posts)
        if len(monthly) >= 2:
            vel = growth_velocity(monthly)
            best, worst = best_and_worst_months(monthly)
            lines.append("\n📈 GROWTH TRENDS")
            lines.append("-" * 40)
            lines.append(f"Best month: {best.get('month')} ({best.get('avg_engagement',0):.0f} avg engagement)")
            lines.append(f"Worst month: {worst.get('month')} ({worst.get('avg_engagement',0):.0f} avg engagement)")
            lines.append("\nMonth-by-month:")
            for _, row in vel.iterrows():
                change = f" ({row['engagement_change_pct']:+.0f}%)" if not pd.isna(row["engagement_change_pct"]) else ""
                lines.append(f"  {row['month']}: {row['posts']} posts, {row['avg_engagement']:.0f} avg eng{change}")
    except Exception:
        pass

    # Caption analysis
    try:
        if "caption" in posts.columns:
            bucket_df = caption_length_buckets(posts)
            best_bucket = bucket_df.loc[bucket_df["avg_engagement"].idxmax()]
            lines.append("\n✍️ CAPTION ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Best caption length: {best_bucket['length_bucket']} ({best_bucket['avg_engagement']:.0f} avg engagement)")
            feat_df = caption_feature_impact(posts)
            for _, row in feat_df.iterrows():
                direction = "✅ helps" if row["lift_pct"] > 5 else ("❌ hurts" if row["lift_pct"] < -5 else "→ neutral")
                lines.append(f"  {row['feature']}: {direction} ({row['lift_pct']:+.0f}%)")
    except Exception:
        pass

    # Competitors
    comp_handles = list_cached_competitors()
    if comp_handles:
        lines.append("\n🔍 COMPETITOR SNAPSHOT")
        lines.append("-" * 40)
        for handle in comp_handles:
            data = load_competitor(handle)
            if data:
                comp = data["posts"].copy()
                comp["engagement"] = comp["likes"] + comp["comments"]
                lines.append(f"@{handle}: {len(comp)} posts, {comp['engagement'].mean():.0f} avg engagement")
                if "content_type" in comp.columns:
                    best = comp.groupby("content_type")["engagement"].mean().idxmax()
                    lines.append(f"  Best format: {best}")

    # AI summary
    if ai_summary:
        lines.append("\n🤖 AI INSIGHTS")
        lines.append("-" * 40)
        lines.append(ai_summary)

    lines.append("\n" + "=" * 60)
    lines.append("Generated by Instagram Growth OS")
    lines.append("=" * 60)

    return "\n".join(lines)


def _build_html_report(text_report: str) -> str:
    escaped = text_report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Colour score badges
    import re
    escaped = re.sub(r"(🔥 Viral|✅ Strong|👍 Average|📉 Weak|❌ Flopped)", r'<span style="color:#E1306C">\1</span>', escaped)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Instagram Growth OS Report</title>
<style>
  body {{ background:#0d0d0d; color:#eee; font-family:monospace; padding:40px; max-width:800px; margin:auto; }}
  pre {{ white-space:pre-wrap; word-break:break-word; line-height:1.6; }}
  h1 {{ color:#E1306C; }}
</style>
</head>
<body>
<h1>📈 Instagram Growth OS — Report</h1>
<pre>{escaped}</pre>
</body>
</html>"""


def render():
    st.title("📥 Export Report")
    st.caption("Download your full Instagram growth analysis as a text or HTML file.")

    cached = load_analysis()
    if not cached:
        st.info("Upload your Instagram export in **My Analytics** first.")
        return

    posts = cached["posts"]
    if posts.empty:
        st.warning("No post data found.")
        return

    st.markdown(f"**Data age:** {cache_age_str()}")
    st.markdown(f"**Posts in report:** {len(posts)}")

    comp_handles = list_cached_competitors()
    if comp_handles:
        st.markdown(f"**Competitors included:** {', '.join('@' + h for h in comp_handles)}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Report (.txt)")
        if st.button("Generate Text Report", type="primary"):
            with st.spinner("Building report..."):
                try:
                    report = _build_text_report(cached)
                    st.download_button(
                        label="Download .txt",
                        data=report,
                        file_name=f"instagram_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                    )
                    with st.expander("Preview report"):
                        st.text(report[:2000] + "\n..." if len(report) > 2000 else report)
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    with col2:
        st.subheader("HTML Report (.html)")
        if st.button("Generate HTML Report"):
            with st.spinner("Building report..."):
                try:
                    text_report = _build_text_report(cached)
                    html = _build_html_report(text_report)
                    st.download_button(
                        label="Download .html",
                        data=html,
                        file_name=f"instagram_report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                    )
                    st.success("HTML report ready — click Download .html above.")
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    st.markdown("---")
    st.subheader("What's in the report?")
    st.markdown("""
- Account overview and date range
- Key metrics (avg engagement, likes, comments)
- Best and worst content types
- Top 5 and bottom 5 posts with scores
- Month-on-month growth trend
- Caption length and feature analysis
- Competitor snapshot (all cached accounts)
- AI insights summary (if generated)
""")
