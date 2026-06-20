# Track-partner integrations: Captur & Cognee

Localhost Search is built so two track-partner tools slot in cleanly. Both share
our core principle: **everything runs on-device, nothing leaves the laptop.**

The adapters live in `integrations/` and are wired into the running service:

| Endpoint | Partner | What it shows |
|---|---|---|
| `GET /trust?path=...` | Captur | on-device trust/quality score for a photo |
| `GET /memory?k=20`    | Cognee | persistent, cross-session search memory |

Each adapter has an `available()` check: if the real partner SDK is installed we
route to it; otherwise a local stand-in keeps the behaviour live so we can demo
today. Swapping to the real product is a **one-function change**.

---

## Captur — on-device photo trust layer

**What Captur is:** the first AI photo trust layer that validates images in real
time, fully on-device (milliseconds, no cloud). Photos never leave the device.

**Why it fits us:** our index is only as good as the photos in it. CLIP will
happily embed blurry, blank, corrupt, or screenshot-of-a-screen images and then
surface them as "matches." Captur is the quality/authenticity gate.

**Where it plugs in (`integrations/captur.py`):**
1. **Index-time gate** — validate every image before embedding; skip or tag
   low-trust ones so the index stays clean.
2. **Search-time signal** — show a trust badge and/or down-rank low-trust photos
   so the agent prefers authentic, high-quality results.
3. **Agent tool** — "is this photo real / good quality?" answered on-device in ms.

**Live today:** `GET /trust?path=<image>` returns `{trust, ok, flags, engine}`.
Our fallback computes sharpness/brightness/resolution with Pillow; Captur
replaces that with real authenticity + tamper + policy checks. `engine` reports
`captur` vs `heuristic-fallback` so it's honest on stage.

**Talking point:** *"Today I gate the index with a quick local heuristic. In
production this `validate_image()` call is Captur — same on-device guarantee,
but real authenticity and tamper detection, so search never surfaces a fake or
junk photo."*

---

## Cognee — persistent, queryable on-device memory

**What Cognee is:** an open-source AI memory engine that turns data into
structured, searchable memory so agents can recall, reason, and improve across
sessions — locally.

**Why it fits us:** right now the index is a flat vector store that forgets
everything between queries. Cognee turns it into an actual agent **memory**.

**Where it plugs in (`integrations/cognee.py`):**
1. **Ingest** — feed file contents + captions + EXIF (when/where) into Cognee so
   it builds a linked memory graph (people, places, time, topics), not isolated
   vectors. This is what makes *"the picture with my cat while I was drunk"*
   resolvable — mood/time/relationship live in the graph.
2. **Recall** — the agent asks Cognee in natural language and gets structured,
   cross-session recall + reasoning instead of nearest-neighbour vectors.
3. **Learn** — every search and the result the user opened is written back, so
   results personalise and improve over time.

**Live today:** each search is logged to memory and exposed at `GET /memory`.
Our fallback is a local JSON store with word-overlap recall; Cognee replaces it
with a real semantic memory graph. `cognee_sdk` reports whether the real engine
is active.

**Talking point:** *"Every query already becomes on-device memory — you can see
it at `/memory`. Drop in Cognee and that flat log becomes a structured memory
graph: the agent reasons over time, place, and people, and gets better the more
you use it — still 100% local."*
