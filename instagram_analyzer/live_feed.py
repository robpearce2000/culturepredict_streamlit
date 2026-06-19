"""
Live feed — scrapes the user's own Instagram profile via Apify and saves
it to the same cache as the ZIP upload path. Allows auto-refresh on a TTL.
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import CACHE_DIR, COMPETITOR_MAX_POSTS

_SETTINGS_PATH = CACHE_DIR / "live_settings.json"


# ── Settings persistence ──────────────────────────────────────────────────────

def save_live_settings(handle: str, refresh_hours: int, auto_refresh: bool):
    CACHE_DIR.mkdir(exist_ok=True)
    data = {
        "handle": handle.lstrip("@"),
        "refresh_hours": refresh_hours,
        "auto_refresh": auto_refresh,
    }
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(data, f)


def load_live_settings() -> dict:
    if not _SETTINGS_PATH.exists():
        return {"handle": "", "refresh_hours": 24, "auto_refresh": False}
    with open(_SETTINGS_PATH) as f:
        return json.load(f)


# ── Staleness check ───────────────────────────────────────────────────────────

def cache_is_stale(refresh_hours: int) -> bool:
    path = CACHE_DIR / "my_account.json"
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(hours=refresh_hours)


def cache_age_hours() -> float:
    path = CACHE_DIR / "my_account.json"
    if not path.exists():
        return 9999.0
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600


# ── Scrape own profile ────────────────────────────────────────────────────────

def scrape_own_profile(handle: str, apify_token: str, max_posts: int = COMPETITOR_MAX_POSTS) -> pd.DataFrame:
    """Scrape own profile via Apify — reuses the same actor as competitor scraping."""
    # Temporarily set the token so competitor.py picks it up
    os.environ["APIFY_API_TOKEN"] = apify_token

    # Re-import to pick up the new env var (competitor.py reads it at module level)
    import importlib
    import competitor
    importlib.reload(competitor)

    posts = competitor.scrape_profile(handle, max_posts=max_posts)
    return posts


def refresh_and_save(handle: str, apify_token: str, max_posts: int = COMPETITOR_MAX_POSTS) -> tuple[pd.DataFrame, str]:
    """Scrape profile and write to the my_account cache. Returns (posts, error_msg)."""
    from cache import save_analysis
    try:
        posts = scrape_own_profile(handle, apify_token, max_posts)
        if posts.empty:
            return pd.DataFrame(), "No posts returned from Apify — check the handle is public and correct."
        # Save with empty followers/profile (not available via scrape)
        profile = {"username": handle.lstrip("@"), "source": "live_scrape"}
        save_analysis(posts, pd.DataFrame(), profile, ai_summary="")
        return posts, ""
    except Exception as e:
        return pd.DataFrame(), str(e)
