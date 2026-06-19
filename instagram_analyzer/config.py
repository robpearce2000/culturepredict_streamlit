from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Apify ─────────────────────────────────────────────────────────────────────
APIFY_PROFILE_ACTOR = "apify/instagram-profile-scraper"
APIFY_HASHTAG_ACTOR = "apify/instagram-hashtag-scraper"
APIFY_RUN_MODE = "waitForFinish"
APIFY_TIMEOUT = 120
APIFY_MAX_POSTS = 50

# ── Instagram export ZIP paths ─────────────────────────────────────────────────
ZIP_POSTS_PATHS = [
    "content/posts_1.json",
    "content/posts_2.json",
    "content/posts_3.json",
    "content/reels.json",
    "content/stories.json",
]
ZIP_FOLLOWERS_PATH = "followers_and_following/followers_1.json"
ZIP_PROFILE_PATH = "personal_information/personal_information.json"

# ── AI ────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
AI_MAX_TOKENS_SUMMARY = 800
AI_MAX_TOKENS_FORMULA = 1000
AI_MAX_TOKENS_IDEAS = 1200
AI_MAX_TOKENS_CAPTION = 500

# ── App limits ────────────────────────────────────────────────────────────────
TOP_POSTS_COUNT = 10
TOP_HASHTAGS_COUNT = 20
CONTENT_IDEAS_COUNT = 10
COMPETITOR_MAX_POSTS = 50
