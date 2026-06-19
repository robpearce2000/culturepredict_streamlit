"""
Competitor scraping via Apify Instagram Profile Scraper.
Actor: apify/instagram-profile-scraper
Requires APIFY_API_TOKEN environment variable.
Falls back to mock data if token not set (for demo/dev).
"""
import os
import re
import requests
import pandas as pd
from datetime import datetime

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN", "")
PROFILE_ACTOR = "apify/instagram-profile-scraper"
HASHTAG_ACTOR = "apify/instagram-hashtag-scraper"


def _run_actor(actor_id: str, input_payload: dict, timeout: int = 120) -> list:
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    resp = requests.post(
        url,
        params={"token": APIFY_TOKEN, "timeout": timeout},
        json=input_payload,
        timeout=timeout + 30,
    )
    resp.raise_for_status()
    return resp.json()


def scrape_profile(handle: str, max_posts: int = 50) -> pd.DataFrame:
    handle = handle.lstrip("@")
    if not APIFY_TOKEN:
        return _mock_profile_data(handle, max_posts)

    items = _run_actor(PROFILE_ACTOR, {"usernames": [handle], "resultsLimit": max_posts})
    rows = []
    for item in items:
        for post in item.get("latestPosts", []):
            rows.append(_parse_post(post, handle))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def scrape_hashtag(tag: str, max_posts: int = 50) -> pd.DataFrame:
    tag = tag.lstrip("#")
    if not APIFY_TOKEN:
        return _mock_hashtag_data(tag, max_posts)

    items = _run_actor(HASHTAG_ACTOR, {"hashtags": [tag], "resultsLimit": max_posts})
    rows = []
    for item in items:
        rows.append(_parse_post(item, f"#{tag}"))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_post(post: dict, handle: str) -> dict:
    ts = post.get("timestamp") or post.get("takenAtTimestamp")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            ts = None
    elif isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts)

    caption = post.get("caption", "") or ""
    hashtags = re.findall(r"#\w+", caption.lower())
    likes = int(post.get("likesCount", 0) or 0)
    comments = int(post.get("commentsCount", 0) or 0)
    video_duration = post.get("videoDuration") or post.get("duration")

    media_type = post.get("type", "").lower()
    product_type = post.get("productType", "")
    if "video" in media_type or post.get("isVideo"):
        content_type = "Reel" if product_type == "clips" else "Video"
    elif post.get("childPosts"):
        content_type = "Carousel"
    else:
        content_type = "Image"

    return {
        "handle": handle,
        "timestamp": ts,
        "content_type": content_type,
        "caption": caption,
        "caption_length": len(caption),
        "hashtags": hashtags,
        "hashtag_count": len(hashtags),
        "likes": likes,
        "comments": comments,
        "engagement": likes + comments,
        "video_duration": video_duration,
        "url": post.get("url", ""),
        "thumbnail": post.get("displayUrl", ""),
    }


def engagement_rate_df(posts: pd.DataFrame, follower_count: int = 10000) -> pd.DataFrame:
    df = posts.copy()
    if follower_count > 0:
        df["eng_rate_pct"] = ((df["engagement"] / follower_count) * 100).round(2)
    else:
        df["eng_rate_pct"] = 0.0
    return df


# ── Mock data (used when no Apify token) ─────────────────────────────────────

def _mock_profile_data(handle: str, n: int) -> pd.DataFrame:
    import random
    from datetime import timedelta
    random.seed(hash(handle) % 9999)
    rows = []
    base = datetime(2024, 1, 1)
    types = ["Reel", "Reel", "Carousel", "Image", "Reel"]
    hooks = [
        "Nobody tells you this about growing on Instagram",
        "Stop doing this if you want more followers",
        "How I gained 10k followers in 30 days",
        "The real reason your Reels aren't getting views",
        "This one tweak doubled my engagement overnight",
        "POV: you finally figured out the algorithm",
        "3 things top creators do that you don't",
    ]
    for i in range(n):
        ctype = random.choice(types)
        likes = random.randint(500, 12000) if ctype == "Reel" else random.randint(100, 3000)
        comments = int(likes * random.uniform(0.02, 0.09))
        hook = random.choice(hooks)
        caption = f"{hook}\n\n#{handle} #instagram #growthhack #contentcreator #viral"
        rows.append({
            "handle": handle,
            "timestamp": base + timedelta(days=i * 3 + random.randint(0, 2), hours=random.randint(7, 21)),
            "content_type": ctype,
            "caption": caption,
            "caption_length": len(caption),
            "hashtags": re.findall(r"#\w+", caption),
            "hashtag_count": 5,
            "likes": likes,
            "comments": comments,
            "engagement": likes + comments,
            "video_duration": random.randint(15, 90) if ctype in ("Reel", "Video") else None,
            "url": f"https://instagram.com/p/demo{i}",
            "thumbnail": "",
        })
    return pd.DataFrame(rows)


def _mock_hashtag_data(tag: str, n: int) -> pd.DataFrame:
    import random
    from datetime import timedelta
    random.seed(hash(tag) % 9999)
    rows = []
    base = datetime(2024, 3, 1)
    for i in range(n):
        likes = random.randint(100, 5000)
        comments = int(likes * random.uniform(0.01, 0.06))
        rows.append({
            "handle": f"user_{i}",
            "timestamp": base + timedelta(days=i),
            "content_type": random.choice(["Reel", "Image", "Carousel"]),
            "caption": f"#{tag} #instagood #explore",
            "caption_length": 30,
            "hashtags": [f"#{tag}", "#instagood", "#explore"],
            "hashtag_count": 3,
            "likes": likes,
            "comments": comments,
            "engagement": likes + comments,
            "video_duration": None,
            "url": "",
            "thumbnail": "",
        })
    return pd.DataFrame(rows)
