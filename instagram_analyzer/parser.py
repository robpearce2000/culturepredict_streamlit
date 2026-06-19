"""
Instagram data export parser.
Instagram export ZIP structure (JSON format):
  content/posts_1.json          — feed posts
  content/reels.json            — reels
  content/stories.json          — stories
  likes/liked_posts.json        — posts you liked
  followers_and_following/followers_1.json
  personal_information/personal_information.json
"""
import zipfile
import json
import pandas as pd
from datetime import datetime
import io


def _read_json(zf: zipfile.ZipFile, path: str):
    try:
        with zf.open(path) as f:
            raw = f.read()
            # Instagram sometimes exports with latin-1 encoding
            try:
                return json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                return json.loads(raw.decode("latin-1"))
    except KeyError:
        return None


def _find(zf: zipfile.ZipFile, keyword: str) -> list[str]:
    return [n for n in zf.namelist() if keyword in n and n.endswith(".json")]


def parse_posts(zf: zipfile.ZipFile) -> pd.DataFrame:
    rows = []

    # Exact known paths first, then fallback search
    search_keys = [
        "content/posts_1", "content/posts_2", "content/posts_3",
        "content/reels", "content/stories",
        "posts_1", "reels_", "stories_",
    ]
    seen_paths = set()
    for key in search_keys:
        for path in _find(zf, key):
            if path not in seen_paths:
                seen_paths.add(path)
                _parse_post_file(zf, path, rows)

    return pd.DataFrame(rows)


def _parse_post_file(zf, path, rows):
    data = _read_json(zf, path)
    if not data:
        return

    # Posts can be a list of post objects or a dict with a key
    if isinstance(data, dict):
        data = data.get("ig_media", data.get("media", [data]))
    if not isinstance(data, list):
        data = [data]

    for item in data:
        media_list = item.get("media", [item])
        for media in media_list:
            uri = media.get("uri", "")
            ts = media.get("creation_timestamp", 0)

            # Caption lives in title field
            caption = media.get("title", "") or ""
            if not isinstance(caption, str):
                caption = ""

            likes = len(item.get("likes", []))
            comments = len(item.get("comments", []))
            content_type = _infer_type(uri, path)

            rows.append({
                "timestamp": datetime.fromtimestamp(ts) if ts else None,
                "caption": caption,
                "content_type": content_type,
                "likes": likes,
                "comments": comments,
                "uri": uri,
            })


def parse_followers(zf: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    # Known paths
    known = [
        "followers_and_following/followers_1.json",
        "followers_and_following/followers.json",
        "connections/followers_and_following/followers_1.json",
    ]
    paths = known + _find(zf, "followers_1") + _find(zf, "followers.json")
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        data = _read_json(zf, path)
        if not data:
            continue
        items = data if isinstance(data, list) else data.get("relationships_followers", [])
        for item in items:
            sl = (item.get("string_list_data") or [{}])
            ts = item.get("timestamp") or (sl[0].get("timestamp", 0) if sl else 0)
            if ts:
                rows.append({"timestamp": datetime.fromtimestamp(int(ts))})
    return pd.DataFrame(rows)


def parse_profile(zf: zipfile.ZipFile) -> dict:
    known = [
        "personal_information/personal_information.json",
        "account_information/personal_information.json",
    ]
    paths = known + _find(zf, "personal_information") + _find(zf, "profile_information")
    for path in paths:
        data = _read_json(zf, path)
        if not data:
            continue
        # Structure varies by export version
        profile = (
            data.get("profile_user")
            or data.get("personal_information")
            or [data]
        )
        if isinstance(profile, list) and profile:
            profile = profile[0]
        info = profile.get("string_map_data", {})
        return {
            "username": info.get("Username", {}).get("value", "Unknown"),
            "name": info.get("Name", {}).get("value", ""),
            "bio": info.get("Bio", {}).get("value", ""),
            "followers": info.get("Followers Count", {}).get("value", "N/A"),
            "following": info.get("Following Count", {}).get("value", "N/A"),
        }
    return {"username": "Unknown", "name": "", "bio": "", "followers": "N/A", "following": "N/A"}


def _infer_type(uri: str, path: str) -> str:
    if "reel" in path.lower() or (uri and uri.endswith(".mp4")):
        return "Reel"
    if "stor" in path.lower():
        return "Story"
    if uri and uri.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return "Image"
    return "Post"


def load_export(uploaded_file) -> dict:
    file_bytes = uploaded_file.read()
    zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    return {
        "posts": parse_posts(zf),
        "followers": parse_followers(zf),
        "profile": parse_profile(zf),
        "file_list": zf.namelist(),
    }
