"""Side-by-side translation server for the quantized GGUF builds.

Serves a phone-friendly page on 0.0.0.0:8735 that translates one input
through several quantizations at once, so the quality difference between
Q4_K_M and Q2_K can be read directly instead of inferred from a chrF++
table. Each build is loaded on demand by llama.cpp (one process per
request, as in eval/eval_gguf.py, because llama-server cannot serve
encoder-decoder models), so nothing sits in VRAM between requests.

Also serves both papers: /paper (quantization) and /paper1 (the original).

  Start-Process .venv\\Scripts\\python.exe scripts\\quant_server.py

API: POST /translate {"text": "...", "models": ["q4_k_m", "q2_k"]} ->
     {"results": [{"model", "dv", "roundtrip_ok", "seconds", "tok_per_sec"}]}
"""

import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keymap import keymap_to_thaana, thaana_to_keymap

PORT = 8735
ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf"
EXE = ROOT / "third_party" / "bin" / "llama-completion.exe"
OUT_DIR = ROOT / "eval" / "outputs"
FONT = ROOT / "paper" / "fonts" / "Faruma.ttf"
PAPER_PDF = ROOT / "paper" / "how-low-can-you-go.pdf"
PAPER_ONE = ROOT / "paper" / "paper.html"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EVAL_TIME = re.compile(r"common_perf_print:\s+eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs")
GEN_LOCK = threading.Lock()


def discover():
    """GGUF builds on disk, annotated with their measured devtest chrF++."""
    scores = {}
    for f in OUT_DIR.glob("gguf-*_devtest.jsonl"):
        tag = f.name[len("gguf-"):-len("_devtest.jsonl")]
        if tag.endswith("-s150") or "parity" in tag:
            continue  # screens and parity checks, not full-set numbers
        try:
            m = json.loads(f.open(encoding="utf-8").readline())["_metrics"]
            scores[tag] = m["chrF++"]
        except Exception:
            pass

    builds = []
    for path in sorted(GGUF_DIR.glob("madlad3b-en-dv-*.gguf"),
                       key=lambda p: -p.stat().st_size):
        name = path.stem.replace("madlad3b-en-dv-", "")
        builds.append({
            "name": name,
            "size_mb": round(path.stat().st_size / 1e6),
            "chrf": scores.get(name),
            "path": str(path),
        })
    return builds


BUILDS = {b["name"]: b for b in discover()}


def translate(build, lines, n_gpu_layers=99):
    cmd_base = [
        str(EXE), "-m", BUILDS[build]["path"], "-n", "256", "-c", "512",
        "--temp", "0", "-ngl", str(n_gpu_layers), "--no-warmup", "-no-cnv",
    ]
    outs, eval_ms, toks = [], 0.0, 0
    t0 = time.time()
    for line in lines:
        proc = subprocess.run(
            cmd_base + ["-p", f"<2dv> {line}"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-500:] or "llama-completion failed")
        outs.append(ANSI.sub("", proc.stdout).replace("[end of text]", "").strip())
        m = EVAL_TIME.search(proc.stderr or "")
        if m:
            eval_ms += float(m.group(1))
            toks += int(m.group(2))
    return {
        "model": build,
        "size_mb": BUILDS[build]["size_mb"],
        "chrf": BUILDS[build]["chrf"],
        "lines": [
            {"dv": o, "roundtrip_ok": keymap_to_thaana(thaana_to_keymap(o)) == o}
            for o in outs
        ],
        "seconds": round(time.time() - t0, 1),
        "tok_per_sec": round(toks / (eval_ms / 1000), 1) if eval_ms else None,
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EN → DV · quantization comparison</title><style>
@font-face{font-family:Faruma;src:url(/font/faruma.ttf) format('truetype')}
:root{color-scheme:dark}
body{background:#111418;color:#e6e6e6;font-family:system-ui,sans-serif;
     max-width:820px;margin:0 auto;padding:1rem 1rem 3rem}
h1{font-size:1rem;color:#8ab4f8;font-weight:600}
a{color:#8ab4f8}
textarea{width:100%;min-height:90px;background:#1b1f26;color:#e6e6e6;
     border:1px solid #333;border-radius:8px;padding:.7rem;font-size:1rem;
     box-sizing:border-box;-webkit-text-size-adjust:100%}
button{background:#8ab4f8;color:#111;border:0;border-radius:8px;
     padding:.7rem 1.5rem;font-size:1rem;font-weight:600;margin-top:.6rem}
button:disabled{opacity:.5}
select{background:#1b1f26;color:#e6e6e6;border:1px solid #333;
     border-radius:8px;padding:.6rem;font-size:1rem;margin-top:.6rem;
     width:100%;box-sizing:border-box}
.card{background:#1b1f26;border:1px solid #2a2f37;border-radius:8px;
     padding:.7rem .9rem;margin-top:.7rem}
.hdr{display:flex;justify-content:space-between;align-items:baseline;
     gap:.5rem;font-size:.8rem;color:#9aa0a6;margin-bottom:.45rem}
.name{color:#8ab4f8;font-weight:600;font-size:.92rem}
.dv{font-family:Faruma,serif;direction:rtl;text-align:right;
     font-size:1.5rem;line-height:2.15;margin:.25rem 0}
.en{color:#9aa0a6;font-size:.8rem;border-top:1px solid #2a2f37;
     padding-top:.4rem;margin-top:.4rem}
.bad{color:#f28b82}.dim{color:#9aa0a6;font-size:.78rem}
</style></head><body>
<h1>English → Dhivehi · compare quantizations
&nbsp;·&nbsp; <a href="/paper">paper</a> <a href="/paper1">paper 1</a></h1>
<textarea id="t" placeholder="Type English. One sentence per line."></textarea>
<select id="model"></select>
<button id="go" onclick="run()">Translate</button>
<div id="out"></div>
<script>
fetch('/builds').then(r=>r.json()).then(d=>{
  document.getElementById('model').innerHTML=d.builds.map(b=>
    `<option value="${b.name}" ${b.name==='q4_k_m'?'selected':''}>
     ${b.name} — ${(b.size_mb/1000).toFixed(2)} GB${
       b.chrf?' — chrF++ '+b.chrf:''}</option>`).join('');
});
function esc(s){return s.replace(/[&<>"]/g,c=>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function run(){
  const btn=document.getElementById('go'),out=document.getElementById('out');
  const text=document.getElementById('t').value.trim();
  const models=[document.getElementById('model').value];
  if(!text){out.innerHTML=
    '<div class="card bad">Enter some text first.</div>';return}
  btn.disabled=true;btn.textContent='Translating…';
  out.innerHTML='<div class="card dim">loading '+models[0]+
    ', a few seconds…</div>';
  try{
    const r=await fetch('/translate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,models})});
    const d=await r.json();
    if(d.error){out.innerHTML='<div class="card bad">'+esc(d.error)+'</div>';return}
    out.innerHTML=d.results.map(m=>`<div class="card">
      <div class="hdr"><span class="name">${esc(m.model)}</span>
      <span>${(m.size_mb/1000).toFixed(2)} GB${
        m.chrf?' · chrF++ '+m.chrf:''} · ${m.seconds}s${
        m.tok_per_sec?' · '+m.tok_per_sec+' tok/s':''}</span></div>
      ${m.lines.map((l,i)=>`<div class="dv">${esc(l.dv)}</div>`+
        (l.roundtrip_ok?'':'<div class="bad dim">⚠ round-trip failed</div>')+
        `<div class="en">${esc(d.source[i])}</div>`).join('')}
      </div>`).join('');
  }catch(e){out.innerHTML='<div class="card bad">'+esc(String(e))+'</div>'}
  btn.disabled=false;btn.textContent='Translate';
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, '{"status":"ready"}')
        elif self.path == "/builds":
            self._send(200, json.dumps({"builds": list(BUILDS.values())}))
        elif self.path == "/font/faruma.ttf" and FONT.exists():
            self._send(200, FONT.read_bytes(), "font/ttf")
        elif self.path == "/paper" and PAPER_PDF.exists():
            self._send(200, PAPER_PDF.read_bytes(), "application/pdf")
        elif self.path == "/paper1" and PAPER_ONE.exists():
            self._send(200, PAPER_ONE.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path != "/translate":
            self._send(404, '{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            lines = [l.strip() for l in req["text"].splitlines() if l.strip()][:8]
            models = [m for m in req.get("models", []) if m in BUILDS][:6]
            if not lines or not models:
                self._send(400, '{"error":"need text and at least one build"}')
                return
            with GEN_LOCK:
                results = [translate(m, lines) for m in models]
            self._send(200, json.dumps(
                {"source": lines, "results": results}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    print(f"builds: {', '.join(BUILDS)}", flush=True)
    print(f"serving on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
