# Localhost Search

On-device, **fully offline** semantic search for your photos and files.
Type what you *remember* ("my cat on the sofa", "night out with drinks",
"whiteboard with diagrams") and get the right photo back — no filenames, no cloud.

Built for the **Localhost: On-Device Agent Hackathon**.

## Why it's an agent (not just search)
- **CLIP** (`clip-ViT-B-32`) embeds images and your text query into one shared space.
- **MiniLM** (`all-MiniLM-L6-v2`) embeds your text files.
- A **local LLM via Ollama** (`llama3.2:3b`) *routes* each query (photos vs files),
  and a **local vision model** (`llava:7b`) *explains why* the top photo matches.
- Everything runs on-device. Flip **Offline mode** (or turn off Wi-Fi) to prove it.

## Setup (do this while online — once)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llava:7b && ollama pull llama3.2:3b
# Cache the CLIP + MiniLM weights:
python -c "import engine; engine.get_clip(); engine.get_text_model()"
```

## Run
```bash
source .venv/bin/activate
streamlit run app.py
```
1. In the sidebar, set the folder to index (e.g. a demo photo folder).
2. Click **Build / rebuild index**.
3. Search.

CLI alternative:
```bash
python index.py /path/to/photos     # build index
```

## Offline demo
- Keep **Offline mode** toggle ON (sets `HF_HUB_OFFLINE=1`).
- For the dramatic version: literally turn off Wi-Fi before searching.
- Make sure `ollama serve` is running locally (it runs offline).

## Spotlight-style overlay (recommended) — native, no browser
A borderless floating search bar that appears on a global hotkey, searches live
as you type, and opens the file on Enter/click. Requires the service running.

```bash
source .venv/bin/activate
python service.py        # terminal 1 (keeps models warm)
python spotlight.py      # terminal 2 (test the overlay)
```

Global hotkey (no Accessibility permission): macOS **Shortcuts** app → new
shortcut → action **Run Shell Script**:

```bash
/Users/gracexu/Projects/localhost-search/.venv/bin/python /Users/gracexu/Projects/localhost-search/spotlight.py
```

Then open the shortcut's **(i) panel → Add Keyboard Shortcut** and pick your combo.

## Fast live web UI — always-on service
Keeps models warm (searches in ~20ms) and serves a search-as-you-type UI with
live thumbnails; click or press Enter to open a file.

```bash
source .venv/bin/activate
python service.py        # then open http://localhost:8765
```

Global hotkey without any Accessibility permission: in the macOS **Shortcuts**
app, make a shortcut whose single action is **Open URLs → http://localhost:8765**,
then assign it a keyboard shortcut. Hotkey → type → Enter opens the top file.

## Use it like a real search bar (menu-bar app + global hotkey)
Make it always-available, Spotlight-style:

```bash
source .venv/bin/activate
python menubar.py
```
- A 🔎 icon appears in the menu bar.
- Press **Ctrl + Option + Space** from any app → type your query → Enter.
- Results open in the Streamlit page (`?q=...` pre-fills and runs the search).
- The menu also has: Search…, Open results page, Rebuild index, Start server, Quit.

**One-time permission for the global hotkey:** macOS must trust the app that
listens for keys. Run `python menubar.py` from your own **Terminal**, then grant:
`System Settings → Privacy & Security → Accessibility → enable Terminal`,
and restart the app. (The "Search…" menu item works even without this.)

For daily use, run it from Terminal so it keeps running after you close the IDE.

## Files
- `engine.py` — embeddings, indexing, search (the core).
- `index.py` — CLI indexer.
- `agent.py` — Ollama routing + vision explanations.
- `app.py` — Streamlit UI (reads `?q=` for the hotkey hand-off).
- `menubar.py` — menu-bar launcher + global hotkey (the "search bar").
- `search_open.py` — find best match for a query and open the file directly
  (call from a macOS Shortcut for a no-permission global hotkey).
