"""Always-on local search service and live shortcut UI."""

from __future__ import annotations

import asyncio
import io
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from alphasearch.config import load_settings
from alphasearch.frontend.adapter import FrontendSearchResponse, search_frontend
from alphasearch.frontend.defaults import configure_qwen_lancedb_defaults
from alphasearch.search.service import SearchContext, create_search_context

PORT = 8765
TEXT_RESULT_LIMIT = 5

_allowed_root: Path | None = None


configure_qwen_lancedb_defaults()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared LanceDB and Qwen handles for the shortcut service.

    Args:
        app: FastAPI application instance.

    Yields:
        Control while the shortcut service is running.
    """
    global _allowed_root

    settings = load_settings()
    context = await asyncio.to_thread(create_search_context, settings)
    row_count = context.store.row_count()
    _allowed_root = settings.data_dir.resolve()

    app.state.search_context = context
    print(
        "Ready. "
        f"db={settings.db_dir} table={settings.table_name} "
        f"embedder={settings.embedder} rows={row_count}. "
        f"Open http://localhost:{PORT}"
    )
    if row_count:
        try:
            await asyncio.to_thread(
                search_frontend,
                "warm up",
                candidate_limit=1,
                context=context,
            )
        except Exception as exc:
            print(f"Warm-up skipped: {exc}")
    yield


app = FastAPI(title="AlphaSearch Shortcut", lifespan=lifespan)


def get_search_context(request: Request) -> SearchContext:
    """Return the shared shortcut search context.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Shared LanceDB and embedder dependencies.
    """
    return request.app.state.search_context


def _empty_response() -> FrontendSearchResponse:
    """Return an empty shortcut frontend response."""
    return {"images": [], "texts": []}


def _is_allowed(path: str) -> bool:
    """Check whether a file path is under the configured data directory.

    Args:
        path: User-provided local file path.

    Returns:
        True when the file is under the configured data directory.
    """
    if _allowed_root is None:
        return False
    try:
        resolved_path = Path(path).resolve()
    except Exception:
        return False
    return resolved_path == _allowed_root or _allowed_root in resolved_path.parents


@app.get("/api/search")
def api_search(
    q: str = "",
    k: int = 9,
    search_context: SearchContext = Depends(get_search_context),
) -> FrontendSearchResponse:
    """Search indexed LanceDB chunks for the shortcut frontend.

    Args:
        q: Natural-language search query.
        k: Maximum number of image results to return.
        search_context: Shared LanceDB and embedder dependencies.

    Returns:
        Frontend-compatible image and text result lists.
    """
    query = q.strip()
    if not query or search_context.store.row_count() == 0:
        return _empty_response()
    return search_frontend(
        query,
        image_limit=k,
        text_limit=TEXT_RESULT_LIMIT,
        context=search_context,
    )


@lru_cache(maxsize=512)
def _thumb_bytes(path: str, size: int, mtime: float) -> bytes:
    """Create cached JPEG thumbnail bytes for a local image.

    Args:
        path: Local image path.
        size: Maximum thumbnail width and height.
        mtime: File modification time used as a cache key.

    Returns:
        Encoded JPEG thumbnail bytes.
    """
    im = Image.open(path)
    im = im.convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.get("/thumb")
def thumb(path: str, size: int = 512):
    """Serve a thumbnail for an indexed local image.

    Args:
        path: Local image path.
        size: Maximum thumbnail width and height.

    Returns:
        JPEG thumbnail response.
    """
    if not _is_allowed(path) or not Path(path).exists():
        raise HTTPException(404)
    try:
        data = _thumb_bytes(path, size, Path(path).stat().st_mtime)
    except Exception as exc:
        raise HTTPException(415) from exc
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@app.post("/open")
def open_file(payload: dict[str, str]) -> dict[str, bool]:
    """Open an indexed local file with the platform default app.

    Args:
        payload: JSON payload containing a local `path`.

    Returns:
        Operation status.
    """
    path = payload.get("path", "")
    if not _is_allowed(path) or not Path(path).exists():
        raise HTTPException(404)
    subprocess.run(["open", path], check=False)
    return {"ok": True}


@app.post("/reveal")
def reveal_file(payload: dict[str, str]) -> dict[str, bool]:
    """Reveal an indexed local file in Finder.

    Args:
        payload: JSON payload containing a local `path`.

    Returns:
        Operation status.
    """
    path = payload.get("path", "")
    if not _is_allowed(path) or not Path(path).exists():
        raise HTTPException(404)
    subprocess.run(["open", "-R", path], check=False)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the shortcut search HTML page."""
    return HTML


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Localhost Search</title>
<style>
  :root{
    --bg:#0b0c10; --panel:#15171f; --panel2:#1c1f2a; --text:#eef1f7;
    --muted:#8b93a7; --accent:#6ea8fe; --accent2:#9d7bff; --good:#3ddc97;
    --shadow:0 18px 60px rgba(0,0,0,.55);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:radial-gradient(1200px 700px at 50% -10%, #1a1d2b 0%, var(--bg) 60%);
    color:var(--text); font:15px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",Segoe UI,Roboto,sans-serif;
    display:flex; flex-direction:column; align-items:center; padding:6vh 16px 40px;
  }
  .wrap{width:min(960px,100%)}
  .brand{display:flex;align-items:center;gap:10px;justify-content:center;margin-bottom:18px;color:var(--muted);font-size:13px;letter-spacing:.3px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 10px var(--good)}
  .searchbar{
    display:flex;align-items:center;gap:12px;background:var(--panel);
    border:1px solid #262a38;border-radius:16px;padding:16px 18px;box-shadow:var(--shadow);
    transition:border-color .2s, box-shadow .2s;
  }
  .searchbar:focus-within{border-color:var(--accent);box-shadow:0 0 0 4px rgba(110,168,254,.12),var(--shadow)}
  .searchbar svg{flex:none;opacity:.7}
  #q{flex:1;background:none;border:none;outline:none;color:var(--text);font-size:22px}
  #q::placeholder{color:#5a6177}
  .spin{width:18px;height:18px;border-radius:50%;border:2px solid #2a2f40;border-top-color:var(--accent);animation:s .7s linear infinite;display:none}
  .spin.on{display:block}
  @keyframes s{to{transform:rotate(360deg)}}
  .hint{color:var(--muted);font-size:12.5px;text-align:center;margin:10px 2px 22px}
  .hint kbd{background:#222634;border:1px solid #313646;border-bottom-width:2px;border-radius:6px;padding:1px 6px;font:inherit;font-size:11px}

  .hero{display:none;gap:18px;background:var(--panel);border:1px solid #262a38;border-radius:18px;
        padding:14px;box-shadow:var(--shadow);margin-bottom:20px;cursor:pointer;
        transition:transform .12s ease, border-color .2s; opacity:0; transform:translateY(8px)}
  .hero.show{display:flex; animation:rise .25s ease forwards}
  .hero:hover{transform:translateY(-2px);border-color:var(--accent)}
  .hero img{width:300px;height:300px;object-fit:cover;border-radius:12px;background:#0c0e14;flex:none}
  .hero .meta{display:flex;flex-direction:column;justify-content:center;min-width:0}
  .hero .badge{align-self:flex-start;background:linear-gradient(90deg,var(--accent),var(--accent2));
               color:#0b0c10;font-weight:700;font-size:11px;padding:3px 9px;border-radius:999px;letter-spacing:.4px;margin-bottom:10px}
  .hero h2{margin:0 0 6px;font-size:20px;font-weight:650;word-break:break-word}
  .hero .sub{color:var(--muted);font-size:13px}
  .hero .cta{margin-top:14px;color:var(--accent);font-size:13px}

  .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px;margin:6px 2px 12px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px;margin-bottom:26px}
  .card{background:var(--panel2);border:1px solid #242838;border-radius:14px;overflow:hidden;cursor:pointer;
        transition:transform .12s ease, border-color .2s, box-shadow .2s; opacity:0; transform:translateY(8px)}
  .card.in{animation:rise .28s ease forwards}
  .card:hover{transform:translateY(-3px);border-color:var(--accent);box-shadow:0 10px 30px rgba(0,0,0,.5)}
  .card img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:#0c0e14}
  .card .cap{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;font-size:11.5px;color:var(--muted)}
  .card .pct{color:var(--good);font-weight:700}
  @keyframes rise{to{opacity:1;transform:none}}

  .files{display:flex;flex-direction:column;gap:8px;margin-bottom:30px}
  .file{display:flex;justify-content:space-between;gap:12px;background:var(--panel2);border:1px solid #242838;
        border-radius:10px;padding:10px 14px;cursor:pointer;transition:border-color .2s}
  .file:hover{border-color:var(--accent)}
  .file .nm{font-weight:600}
  .file .sn{color:var(--muted);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60ch}
  .empty{color:#566;text-align:center;margin-top:40px;font-size:14px}
  .toast{position:fixed;bottom:26px;left:50%;transform:translateX(-50%) translateY(20px);
         background:var(--good);color:#03261a;font-weight:700;padding:10px 18px;border-radius:999px;
         opacity:0;transition:.25s;pointer-events:none}
  .toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
</style>
</head>
<body>
  <div class="wrap">
    <div class="brand"><span class="dot"></span> On-device · offline · private</div>
    <div class="searchbar">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#8b93a7" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
      <input id="q" autofocus autocomplete="off" placeholder="Search your photos & files by meaning…"/>
      <div class="spin" id="spin"></div>
    </div>
    <div class="hint">Type to search live · <kbd>↵</kbd> opens the top match · click any result to open it</div>

    <div class="hero" id="hero"></div>
    <div id="more"></div>
    <div id="files"></div>
    <div class="empty" id="empty">Start typing to find anything on your device.</div>
  </div>
  <div class="toast" id="toast"></div>

<script>
const qEl=document.getElementById('q'), spin=document.getElementById('spin');
const hero=document.getElementById('hero'), more=document.getElementById('more');
const filesEl=document.getElementById('files'), empty=document.getElementById('empty');
const toast=document.getElementById('toast');
let top=null, timer=null, ctrl=null, seq=0;

const thumb=(p,s)=>`/thumb?size=${s}&path=`+encodeURIComponent(p);
const pct=v=>Math.round(Math.max(0,Math.min(1,v))*100);

function showToast(t){toast.textContent=t;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),1100);}
async function openFile(p){showToast('Opening…');try{await fetch('/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})});}catch(e){}}

function render(data){
  const imgs=data.images||[], txts=data.texts||[];
  empty.style.display=(imgs.length||txts.length)?'none':'block';
  // hero
  if(imgs.length){
    top=imgs[0];
    hero.innerHTML=`<img src="${thumb(top.path,600)}"/>
      <div class="meta"><span class="badge">BEST MATCH · ${pct(top.score)}%</span>
      <h2>${top.name}</h2>
      <div class="sub">${top.taken_at?('📅 '+top.taken_at.slice(0,10)+' · '):''}${top.path}</div>
      <div class="cta">↵ or click to open in Preview →</div></div>`;
    hero.classList.add('show'); hero.onclick=()=>openFile(top.path);
  } else { hero.classList.remove('show'); hero.innerHTML=''; top=null; }
  // more images
  const rest=imgs.slice(1);
  more.innerHTML = rest.length? `<div class="label">More matches</div><div class="grid">`+
    rest.map((r,i)=>`<div class="card" style="animation-delay:${i*30}ms" onclick='openFile(${JSON.stringify(r.path)})'>
      <img loading="lazy" src="${thumb(r.path,300)}"/>
      <div class="cap"><span>${r.name.length>16?r.name.slice(0,15)+'…':r.name}</span><span class="pct">${pct(r.score)}%</span></div>
    </div>`).join('')+`</div>` : '';
  setTimeout(()=>document.querySelectorAll('.card').forEach(c=>c.classList.add('in')),0);
  // text files
  filesEl.innerHTML = txts.length? `<div class="label">Files</div><div class="files">`+
    txts.map(r=>`<div class="file" onclick='openFile(${JSON.stringify(r.path)})'>
      <div><div class="nm">${r.name}</div><div class="sn">${(r.snippet||'').slice(0,90)}</div></div>
      <div class="pct" style="color:var(--good);font-weight:700">${pct(r.score)}%</div>
    </div>`).join('')+`</div>` : '';
}

async function run(){
  const q=qEl.value.trim();
  if(!q){hero.classList.remove('show');more.innerHTML='';filesEl.innerHTML='';empty.style.display='block';spin.classList.remove('on');return;}
  const my=++seq; spin.classList.add('on');
  if(ctrl) ctrl.abort(); ctrl=new AbortController();
  try{
    const r=await fetch('/api/search?k=9&q='+encodeURIComponent(q),{signal:ctrl.signal});
    const data=await r.json();
    if(my===seq){render(data);}
  }catch(e){}
  finally{ if(my===seq) spin.classList.remove('on'); }
}

qEl.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(run,110);});
qEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&top){openFile(top.path);}});
// prefill from ?q=
const pre=new URLSearchParams(location.search).get('q');
if(pre){qEl.value=pre; run();}
qEl.focus();
</script>
</body>
</html>
"""


def main() -> None:
    """Run the shortcut search UI server."""
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
