import json
import os
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _posts_hash(posts: pd.DataFrame) -> str:
    key = str(len(posts)) + str(posts["timestamp"].min()) + str(posts["timestamp"].max())
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_analysis(posts: pd.DataFrame, followers: pd.DataFrame, profile: dict, ai_summary: str = ""):
    cache = {
        "saved_at": datetime.now().isoformat(),
        "profile": profile,
        "post_count": len(posts),
        "follower_count": len(followers),
        "ai_summary": ai_summary,
        "posts": posts.to_dict(orient="records"),
        "followers": followers.to_dict(orient="records"),
    }
    path = CACHE_DIR / "my_account.json"
    with open(path, "w") as f:
        json.dump(cache, f, default=str)


def load_analysis() -> dict | None:
    path = CACHE_DIR / "my_account.json"
    if not path.exists():
        return None
    with open(path) as f:
        cache = json.load(f)
    cache["posts"] = pd.DataFrame(cache["posts"])
    cache["followers"] = pd.DataFrame(cache["followers"])
    if "timestamp" in cache["posts"].columns:
        cache["posts"]["timestamp"] = pd.to_datetime(cache["posts"]["timestamp"], errors="coerce")
    if "timestamp" in cache["followers"].columns:
        cache["followers"]["timestamp"] = pd.to_datetime(cache["followers"]["timestamp"], errors="coerce")
    return cache


def cache_age_str() -> str:
    path = CACHE_DIR / "my_account.json"
    if not path.exists():
        return "No cached data"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    delta = datetime.now() - mtime
    if delta.days > 0:
        return f"Last updated {delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"Last updated {hours}h ago"
    return f"Last updated {delta.seconds // 60}m ago"


def save_competitor(handle: str, posts: pd.DataFrame, summary: str = ""):
    data = {
        "saved_at": datetime.now().isoformat(),
        "handle": handle,
        "post_count": len(posts),
        "summary": summary,
        "posts": posts.to_dict(orient="records"),
    }
    path = CACHE_DIR / f"competitor_{handle.lstrip('@')}.json"
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def load_competitor(handle: str) -> dict | None:
    path = CACHE_DIR / f"competitor_{handle.lstrip('@')}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data["posts"] = pd.DataFrame(data["posts"])
    if "timestamp" in data["posts"].columns:
        data["posts"]["timestamp"] = pd.to_datetime(data["posts"]["timestamp"], errors="coerce")
    return data


def list_cached_competitors() -> list[str]:
    return [p.stem.replace("competitor_", "@") for p in CACHE_DIR.glob("competitor_*.json")]
