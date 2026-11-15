"""Read-only localhost dashboard. Not exposed on a public network."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from .storage import LocalStore

PAGE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Continual AI Lab</title><style>
body{font-family:system-ui,sans-serif;background:#0b1220;color:#ecf2ff;margin:0;padding:24px}
main{max-width:900px;margin:auto}h1{font-size:2rem}
.panel{background:#161f32;border:1px solid #344158;border-radius:18px;padding:24px;margin:18px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.metric{font-size:2.2rem;font-weight:750;color:#89cbff}small,p{color:#b7c6dc}
code{color:#a1eab5}button{padding:10px 14px;cursor:pointer}
</style></head><body><main><h1>Continual AI — Local Research Lab</h1>
<p>Independent verification. Model-weight learning requires a separate training job.</p>
<div class='panel'><div class='cards'>
<div><small>Attempts</small><div class='metric' id='total'>—</div></div>
<div><small>Verified successes</small><div class='metric' id='correct'>—</div></div>
<div><small>Historical task accuracy</small><div class='metric' id='rate'>—</div></div>
</div></div><div class='panel'><h2>Commands</h2>
<p><code>python -m continual_ai.cli chat</code> — Talk to the model</p>
<p><code>python -m continual_ai.cli loop</code> — Run until Ctrl+C</p>
<p><code>python -m continual_ai.cli benchmark</code> — Independent test</p>
<p><code>python -m continual_ai.cli export</code> — Verified training dataset</p>
<p><code>python -m continual_ai.train_lora --help</code> — Optional real LoRA training</p>
</div><p>Stats refresh every 5 seconds. Dashboard bound to 127.0.0.1.</p></main>
<script>async function load(){try{let r=await fetch('/api/stats');let s=await r.json();
for(let k of ['total','correct'])document.getElementById(k).textContent=s[k];
document.getElementById('rate').textContent=s.accuracy===null?'—':(100*s.accuracy).toFixed(1)+'%';
}catch(e){console.error(e)}}load();setInterval(load,5000);</script></body></html>"""


def serve(db_path: Path, port: int = 8765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/stats":
                db = LocalStore(db_path)
                try:
                    payload = json.dumps(db.stats()).encode("utf-8")
                finally:
                    db.close()
                mimetype = "application/json"
            elif self.path in ("/", "/index.html"):
                payload = PAGE.encode("utf-8")
                mimetype = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Dashboard: http://127.0.0.1:{port} (Ctrl+C to stop)")
        server.serve_forever()
