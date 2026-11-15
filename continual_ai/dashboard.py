"""Read-only local dashboard with re-scored answers and transparent training state."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from .storage import LocalStore

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continual AI Lab — Diagnostics</title><style>
:root{color-scheme:dark}body{font-family:system-ui,Arial,sans-serif;background:#0b1220;color:#ecf2ff;margin:0;padding:24px}
main{max-width:1200px;margin:auto}h1{font-size:2rem}h2{font-size:1.2rem}
.panel{background:#161f32;border:1px solid #344158;border-radius:17px;padding:22px;margin:18px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px}
.metric{font-size:2.1rem;font-weight:750;color:#9ecaff}small,p,.help{color:#b7c6dc}
code{color:#a1eab5}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #35445f;padding:10px;text-align:left;vertical-align:top}th{color:#9ecaff}td.answer{font-family:monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-width:320px}
.ok{color:#8fe7b0}.fail{color:#ffaeaa}.muted{color:#b7c6dc} .scroll{overflow-x:auto}a{color:#9ecaff}
</style></head><body><main><h1>Continual AI — Local Research Lab</h1>
<p>Mathematical results are independently re-scored against a deterministic oracle. Formatting errors and neural-model training are tracked separately.</p>
<div class="panel"><div class="cards">
<div><small>Learning attempts</small><div class="metric" id="total">—</div></div>
<div><small>Re-scored correct</small><div class="metric" id="correct">—</div></div>
<div><small>Re-scored accuracy</small><div class="metric" id="rate">—</div></div>
<div><small>Last 100 accuracy</small><div class="metric" id="recent">—</div></div>
<div><small>Recovered formatting successes</small><div class="metric" id="rescued">—</div></div>
<div><small>Unparseable responses</small><div class="metric" id="unparseable">—</div></div>
<div><small>Verifier inference calls</small><div class="metric" id="verify">—</div></div>
<div><small>Verifier inference time</small><div class="metric" id="time">—</div></div>
</div><p class="help" id="legacy">Original recorded results: —. No local database records have been changed.</p></div>
<div class="panel"><h2>Model training / deployment</h2><p id="trainer">Checking local training status...</p>
<p id="adapter">No adapter registered.</p><p class="help">The Ollama Qwen3-4B learning agent does NOT automatically use a promoted Qwen3-1.7B LoRA adapter. Adapter promotion is validated only on a narrow arithmetic benchmark.</p></div>
<div class="panel"><h2>Performance by task family</h2><div class="scroll"><table><thead><tr><th>Family</th><th>Correct</th><th>Attempts</th><th>Accuracy</th></tr></thead><tbody id="families"></tbody></table></div></div>
<div class="panel"><h2>Latest learning responses</h2><p class="help">The raw model response and parser method are shown so a formatting problem cannot be mistaken for weak reasoning.</p>
<div class="scroll"><table><thead><tr><th>Seed</th><th>Family</th><th>Expected</th><th>Model response</th><th>Parser</th><th>Result</th></tr></thead><tbody id="attempts"></tbody></table></div></div>
<p>Refreshing every 5 seconds. Read-only dashboard bound to 127.0.0.1.</p></main><script>
const $=id=>document.getElementById(id);const pct=n=>n==null?'—':(100*n).toFixed(1)+'%';
function cell(row,text,cls=''){let c=document.createElement('td');c.textContent=String(text??'');if(cls)c.className=cls;row.appendChild(c)}
async function refresh(){try{const res=await fetch('/api/stats',{cache:'no-store'});if(!res.ok)throw new Error('HTTP '+res.status);const s=await res.json();
$('total').textContent=s.total;$('correct').textContent=s.correct;$('rate').textContent=pct(s.accuracy);
$('recent').textContent=pct(s.recent_accuracy);$('rescued').textContent=s.rescued_from_format;
$('unparseable').textContent=s.unparseable;$('verify').textContent=s.verifier_share==null?'—':pct(s.verifier_share);
$('time').textContent=pct(s.verifier_time_share);
$('legacy').textContent='Original logged successes: '+s.legacy_correct+'. Re-scored results are calculated without rewriting logs. Invalid stored task labels: '+s.invalid_stored_labels+'.';
$('trainer').textContent='Training status: '+(s.training_running?'running':(s.last_training_outcome||'not yet recorded'));
$('adapter').textContent=s.active_adapter?('Promoted model: '+s.active_base_model+' | adapter path: '+s.active_adapter):'No locally promoted adapter.';
const kinds=$('families');kinds.replaceChildren();for(let [name,v] of Object.entries(s.kinds||{})){let r=document.createElement('tr');cell(r,name);cell(r,v.passed);cell(r,v.total);cell(r,pct(v.total?v.passed/v.total:null));kinds.appendChild(r)}
const attempts=$('attempts');attempts.replaceChildren();for(let a of s.recent_attempts||[]){let r=document.createElement('tr');cell(r,a.seed);cell(r,a.kind);cell(r,a.expected);cell(r,a.response,'answer');cell(r,a.parse_method);cell(r,a.correct?'PASS':'FAIL',a.correct?'ok':'fail');attempts.appendChild(r)}
}catch(e){$('trainer').textContent='Dashboard stats unavailable: '+e.message;console.error(e)}}refresh();setInterval(refresh,5000);
</script></body></html>"""


def serve(db_path: Path, port: int = 8765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/stats':
                db = LocalStore(db_path)
                try:
                    stats = db.stats()
                    stats['recent_attempts'] = db.recent_attempts(12)
                    agent_file = db_path.parent / 'agent_state.json'
                    try:
                        agent = json.loads(agent_file.read_text(encoding='utf-8')) if agent_file.is_file() else {}
                    except (json.JSONDecodeError, OSError):
                        agent = {}
                    calls = agent.get('calls', {})
                    total = sum(calls.values())
                    stats['verifier_share'] = calls.get('verifier', 0) / total if total else None
                    elapsed = agent.get('seconds', {})
                    elapsed_total = sum(elapsed.values())
                    stats['verifier_time_share'] = elapsed.get('verifier', 0) / elapsed_total if elapsed_total else None
                    from .training_registry import read_active
                    try:
                        active = read_active(db_path.parent / 'training')
                    except (OSError, ValueError, json.JSONDecodeError):
                        active = None
                    stats['active_adapter'] = active.get('adapter') if active else None
                    stats['active_base_model'] = active.get('base_model') if active else None
                    train_dir = db_path.parent / 'training'
                    state_file = train_dir / 'train_state.json'
                    try:
                        state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.is_file() else {}
                    except (json.JSONDecodeError, OSError):
                        state = {}
                    stats['last_training_outcome'] = state.get('last_outcome')
                    stats['training_running'] = (db_path.parent / 'training.lock').exists()
                    payload = json.dumps(stats).encode('utf-8')
                finally:
                    db.close()
                mimetype = 'application/json'
            elif self.path in ('/', '/index.html'):
                payload = PAGE.encode('utf-8')
                mimetype = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        print(f'Dashboard: http://127.0.0.1:{port} (Ctrl+C to stop)')
        server.serve_forever()
