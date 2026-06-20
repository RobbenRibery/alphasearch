"""
Localhost Search -- on-device, offline semantic search for your photos & files.

Run:
    streamlit run app.py

Everything runs locally. Flip the "Offline mode" toggle (sets HF_HUB_OFFLINE=1)
to prove there is zero network access during your demo.
"""

import os
from pathlib import Path

import streamlit as st

import engine
import agent

st.set_page_config(page_title="Localhost Search", page_icon="🔎", layout="wide")

# --- Sidebar: index controls -------------------------------------------------
st.sidebar.title("🔎 Localhost Search")
st.sidebar.caption("On-device · offline · private")

offline = st.sidebar.toggle("Offline mode (HF_HUB_OFFLINE=1)", value=True)
os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"

index_dir = engine.DEFAULT_INDEX_DIR

folder = st.sidebar.text_input(
    "Folder to index", value=str(Path.home() / "Documents" / "demo_files")
)
if st.sidebar.button("Build / rebuild index", type="primary"):
    prog = st.sidebar.progress(0.0, text="Starting...")

    def cb(stage, done, total):
        prog.progress(min(done / max(total, 1), 1.0), text=f"{stage}: {done}/{total}")

    with st.spinner("Indexing on-device..."):
        stats = engine.build_index(folder, index_dir, progress=cb)
    st.sidebar.success(f"Indexed {stats['num_images']} images, {stats['num_texts']} files")

use_agent = st.sidebar.checkbox("Use local LLM agent (Ollama)", value=agent.available())
if use_agent and not agent.available():
    st.sidebar.warning("Ollama not reachable — falling back to keyword routing.")


@st.cache_resource(show_spinner="Warming up local models...")
def _warm():
    return agent.warm_up()


if use_agent and agent.available():
    _warm()

# --- Load index --------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_index(_dir):
    return engine.SearchIndex(_dir)


idx = None
if (Path(index_dir) / "image_meta.json").exists():
    try:
        idx = load_index(index_dir)
        st.sidebar.info(
            f"Index: {idx.info['num_images']} images · {idx.info['num_texts']} files\n\n"
            f"from {idx.info['root']}"
        )
    except Exception as e:
        st.sidebar.error(f"Could not load index: {e}")

# --- Main: search ------------------------------------------------------------
st.title("Search your stuff by meaning")
st.caption(
    "Type what you remember, not the filename. "
    "e.g. *“my cat sleeping on the sofa”*, *“night out with drinks”*, *“whiteboard with diagrams”*"
)

prefill = st.query_params.get("q", "")
query = st.text_input(
    "What are you looking for?", value=prefill, placeholder="my cat on the sofa"
)

if query and idx is None:
    st.warning("Build an index first (sidebar).")

if query and idx is not None:
    mode = "both"
    if use_agent:
        with st.spinner("Agent deciding where to look..."):
            mode = agent.route(query)
        st.caption(f"🧭 Agent routed this to: **{mode}**")

    if mode in ("images", "both"):
        img_results = idx.search_images(query, k=12)
        if img_results:
            top = img_results[0]
            st.subheader("📷 Best match")
            # Large hero image, centered and width-capped so portrait shots
            # stay big but not absurdly tall.
            hero_l, hero_c, hero_r = st.columns([1, 2, 1])
            with hero_c:
                try:
                    st.image(top["path"], use_container_width=True)
                except Exception:
                    st.write(Path(top["path"]).name)
                cap = f"score {top['score']:.2f}"
                if top.get("taken_at"):
                    cap += f" · {top['taken_at'][:10]}"
                st.caption(cap)

            if use_agent and agent.available():
                with st.spinner("Vision model explaining the top match..."):
                    why = agent.explain(query, top["path"])
                if why:
                    st.success(f"**Why this matches:** {why}")

            rest = img_results[1:]
            if rest:
                st.caption("Other matches")
                cols = st.columns(6)
                for i, r in enumerate(rest):
                    with cols[i % 6]:
                        try:
                            st.image(r["path"], use_container_width=True)
                        except Exception:
                            st.write(Path(r["path"]).name)
                        st.caption(f"{r['score']:.2f}")

    if mode in ("texts", "both"):
        txt_results = idx.search_texts(query, k=6)
        if txt_results:
            st.subheader("📄 Files")
            for r in txt_results:
                with st.expander(f"{Path(r['path']).name}  ·  score {r['score']:.2f}"):
                    st.write(r["snippet"])
                    st.caption(r["path"])
