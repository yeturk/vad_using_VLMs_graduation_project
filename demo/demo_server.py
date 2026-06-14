"""Local demo server for the VAD project (live + cached inference).

Run inside the grad2_env (so live API calls work), e.g. via:
    wsl -d Ubuntu-22.04 -- bash vera_lite/run_wsl.sh demo/demo_server.py
then open http://localhost:8000 in your browser.

Endpoints:
    GET  /                      -> the demo UI (demo/index.html)
    GET  /api/clips             -> list of clips (id, label, expected, has_cached)
    GET  /api/run?id=&mode=     -> inference result (mode = cached | live)
    GET  /video/<id>            -> the clip's mp4 (with HTTP range support)
"""
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "vera_lite" / "dataset_manifest.json"
QUESTIONS = ROOT / "vera_lite" / "guiding_questions_v3.json"
CACHE_DIR = ROOT / "vera_lite" / "runs" / "blind" / "20260609_121427"  # v3 12/13 run
MODEL = os.getenv("DEMO_MODEL", "qwen3.6-plus")
PORT = int(os.getenv("DEMO_PORT", "8000"))

_live_lock = threading.Lock()


def load_clips():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = []
    for item in manifest:
        cid = item["id"]
        cached = (CACHE_DIR / f"{cid}_learner.json").exists()
        clips.append({
            "id": cid,
            "label": cid.replace("_", " ").title(),
            "expected": item["expected"],
            "anomaly_type": item.get("anomaly_type"),
            "video": item["video"],
            "has_cached": cached,
        })
    return clips


def clip_by_id(cid):
    for c in load_clips():
        if c["id"] == cid:
            return c
    return None


def run_cached(cid):
    path = CACHE_DIR / f"{cid}_learner.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"parsed": data.get("parsed", {}), "questions": data.get("questions", []),
            "source": "cached"}


def run_live(clip):
    from vera_lite.run_learner import run_learner
    with _live_lock:
        result = run_learner(
            video=clip["video"], expected=None, model=MODEL,
            questions_path=QUESTIONS, thinking_budget=2000, max_tokens=3000,
        )
    return {"parsed": result.get("parsed", {}), "questions": result.get("questions", []),
            "source": "live", "usage": result.get("usage", {})}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _video(self, path: Path):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        with open(path, "rb") as f:
            if rng and rng.startswith("bytes="):
                s, _, e = rng[6:].partition("-")
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                f.seek(start)
                chunk = f.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                self.wfile.write(f.read())

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/" or u.path == "/index.html":
                return self._file(HERE / "index.html", "text/html; charset=utf-8")
            if u.path == "/api/clips":
                return self._json({"clips": load_clips(), "model": MODEL})
            if u.path == "/api/run":
                cid = (q.get("id") or [""])[0]
                mode = (q.get("mode") or ["cached"])[0]
                clip = clip_by_id(cid)
                if not clip:
                    return self._json({"error": f"unknown clip {cid}"}, 404)
                if mode == "live":
                    return self._json(run_live(clip))
                if not clip["has_cached"]:
                    return self._json({"error": "no cached result"}, 404)
                return self._json(run_cached(cid))
            if u.path.startswith("/video/"):
                cid = u.path[len("/video/"):]
                clip = clip_by_id(cid)
                if not clip:
                    return self._json({"error": "unknown clip"}, 404)
                vpath = ROOT / clip["video"]
                if not vpath.exists():
                    return self._json({"error": "video missing"}, 404)
                return self._video(vpath)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # never crash the demo
            self._json({"error": str(e)}, 500)


if __name__ == "__main__":
    print(f"Demo server: http://localhost:{PORT}  (model={MODEL})")
    print(f"Cached results: {CACHE_DIR}  (exists={CACHE_DIR.exists()})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
