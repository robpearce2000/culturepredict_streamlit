import anthropic
import pandas as pd

from config import CLAUDE_MODEL, AI_MAX_TOKENS_IDEAS, AI_MAX_TOKENS_CAPTION, CONTENT_IDEAS_COUNT
from prompts import content_ideas as ideas_prompt, caption_writer


def generate_ideas(
    my_posts: pd.DataFrame,
    competitor_posts: pd.DataFrame,
    api_key: str,
    niche: str = "",
    n_ideas: int = CONTENT_IDEAS_COUNT,
) -> str:
    my_top = ""
    if not my_posts.empty:
        top = my_posts.nlargest(5, "engagement")[["content_type", "caption", "engagement"]]
        my_top = f"My top 5 posts:\n{top.to_string(index=False)}"

    comp_top = ""
    if not competitor_posts.empty:
        top_c = competitor_posts.nlargest(5, "engagement")[["handle", "content_type", "caption", "engagement"]]
        comp_top = f"Top competitor posts:\n{top_c.to_string(index=False)}"

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_IDEAS,
        messages=[{
            "role": "user",
            "content": ideas_prompt(niche, my_top, comp_top, n_ideas),
        }],
    )
    return message.content[0].text


def generate_caption(post_idea: str, api_key: str, style: str = "conversational") -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=AI_MAX_TOKENS_CAPTION,
        messages=[{"role": "user", "content": caption_writer(post_idea, style)}],
    )
    return message.content[0].text
