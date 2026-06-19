"""
Hook Library — extracts opening lines from competitor top posts and saves them
as a swipe file you can reference when writing your own content.
"""
import json
import re
import anthropic
import pandas as pd
from pathlib import Path

from config import CACHE_DIR, CLAUDE_MODEL, AI_MAX_TOKENS_FORMULA

HOOKS_FILE = CACHE_DIR / "hook_library.json"


def _load_hooks() -> list[dict]:
    if HOOKS_FILE.exists():
        with open(HOOKS_FILE) as f:
            return json.load(f)
    return []


def _save_hooks(hooks: list[dict]):
    with open(HOOKS_FILE, "w") as f:
        json.dump(hooks, f, indent=2, default=str)


def extract_hooks_from_posts(posts: pd.DataFrame, handle: str, api_key: str) -> list[dict]:
    """AI reads top posts and extracts the hook (first line / opening angle)."""
    top = posts.nlargest(15, "engagement")
    captions = []
    for _, row in top.iterrows():
        caption = str(row.get("caption", ""))[:300]
        captions.append(f"[Engagement: {row['engagement']:,}] {caption}")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_FORMULA,
        messages=[{
            "role": "user",
            "content": f"""Extract the hooks from @{handle}'s top posts. A hook is the first line or opening angle that grabs attention.

Posts (sorted by engagement):
{chr(10).join(captions)}

For each post, output exactly this format on one line:
HOOK: [the hook text] | TYPE: [Question/Shock/Story/Proof/Controversy/Curiosity] | ENGAGEMENT: [number]

Extract one hook per post. Output nothing else."""
        }]
    )

    hooks = []
    existing = {h["hook"] for h in _load_hooks()}

    for line in message.content[0].text.strip().split("\n"):
        if not line.startswith("HOOK:"):
            continue
        try:
            parts = dict(p.split(": ", 1) for p in line.split(" | "))
            hook_text = parts.get("HOOK", "").strip()
            if hook_text and hook_text not in existing:
                hooks.append({
                    "hook": hook_text,
                    "type": parts.get("TYPE", "Unknown").strip(),
                    "engagement": parts.get("ENGAGEMENT", "0").strip(),
                    "source": f"@{handle}",
                    "saved": True,
                })
        except Exception:
            continue

    return hooks


def add_to_library(hooks: list[dict]):
    existing = _load_hooks()
    existing_texts = {h["hook"] for h in existing}
    new_hooks = [h for h in hooks if h["hook"] not in existing_texts]
    _save_hooks(existing + new_hooks)
    return len(new_hooks)


def get_library() -> pd.DataFrame:
    hooks = _load_hooks()
    if not hooks:
        return pd.DataFrame()
    return pd.DataFrame(hooks)


def delete_hook(hook_text: str):
    hooks = [h for h in _load_hooks() if h["hook"] != hook_text]
    _save_hooks(hooks)


def get_hooks_by_type(hook_type: str) -> pd.DataFrame:
    df = get_library()
    if df.empty:
        return df
    return df[df["type"].str.lower() == hook_type.lower()]
