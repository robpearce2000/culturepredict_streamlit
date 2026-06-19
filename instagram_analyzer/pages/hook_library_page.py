import streamlit as st
import pandas as pd

from cache import list_cached_competitors, load_competitor
from hook_library import (
    extract_hooks_from_posts, add_to_library,
    get_library, delete_hook, get_hooks_by_type
)

HOOK_TYPES = ["All", "Question", "Shock", "Story", "Proof", "Controversy", "Curiosity"]


def render():
    st.title("🪝 Hook Library")
    st.caption("A swipe file of proven opening lines extracted from your competitors' best posts. Use these as starting points for your own content.")

    api_key = st.session_state.get("api_key", "")

    # ── Extract new hooks ─────────────────────────────────────────────────────
    with st.expander("➕ Extract hooks from a competitor", expanded=get_library().empty):
        competitors = list_cached_competitors()
        if not competitors:
            st.info("Analyse a competitor in **Competitor Spy** first.")
        elif not api_key:
            st.warning("Add your Claude API key in ⚙️ Settings.")
        else:
            handle = st.selectbox("Choose competitor", competitors, key="hook_comp").lstrip("@")
            if st.button("Extract Hooks", type="primary"):
                data = load_competitor(handle)
                if data:
                    with st.spinner(f"Extracting hooks from @{handle}'s top posts..."):
                        try:
                            hooks = extract_hooks_from_posts(data["posts"], handle, api_key)
                            added = add_to_library(hooks)
                            st.success(f"Added {added} new hooks to your library.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")

    st.markdown("---")

    # ── Library view ──────────────────────────────────────────────────────────
    library = get_library()

    if library.empty:
        st.info("Your hook library is empty. Extract hooks from a competitor above to get started.")
        return

    # Filter by type
    col_filter, col_count = st.columns([3, 1])
    with col_filter:
        selected_type = st.selectbox("Filter by hook type", HOOK_TYPES)
    with col_count:
        st.metric("Total hooks", len(library))

    if selected_type != "All":
        filtered = library[library["type"].str.lower() == selected_type.lower()]
    else:
        filtered = library

    if filtered.empty:
        st.info(f"No {selected_type} hooks saved yet.")
        return

    # Sort options
    sort_by = st.radio("Sort by", ["Engagement", "Source", "Type"], horizontal=True)
    if sort_by == "Engagement":
        filtered = filtered.copy()
        filtered["_eng_sort"] = pd.to_numeric(
            filtered["engagement"].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
        filtered = filtered.sort_values("_eng_sort", ascending=False).drop(columns=["_eng_sort"])
    elif sort_by == "Source":
        filtered = filtered.sort_values("source")
    elif sort_by == "Type":
        filtered = filtered.sort_values("type")

    st.markdown("---")

    # Display hooks as cards
    for _, row in filtered.iterrows():
        col_hook, col_meta, col_del = st.columns([5, 2, 1])
        with col_hook:
            st.markdown(f"**{row['hook']}**")
        with col_meta:
            st.caption(f"{row['type']} · {row['source']} · {row['engagement']} eng")
        with col_del:
            if st.button("🗑", key=f"del_{row['hook'][:20]}", help="Remove from library"):
                delete_hook(row["hook"])
                st.rerun()
        st.markdown("---")

    # ── Add your own hook ─────────────────────────────────────────────────────
    with st.expander("✍️ Add your own hook manually"):
        custom_hook = st.text_input("Hook text")
        custom_type = st.selectbox("Type", HOOK_TYPES[1:], key="custom_type")
        if st.button("Add to Library") and custom_hook:
            added = add_to_library([{
                "hook": custom_hook,
                "type": custom_type,
                "engagement": "manual",
                "source": "manual",
            }])
            if added:
                st.success("Added to library.")
                st.rerun()
            else:
                st.info("That hook is already in your library.")
