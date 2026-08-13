"""Hand-testing inference server for the fine-tuned EN->DV model.

Serves a single-page UI on 0.0.0.0:8734 (LAN + Tailscale) with a POST
/translate API. Loads madlad400-3b-mt via madlad_loader (transformers 5.x
wiring fix), applies the LoRA adapter, and merges it for fast inference.
Every output is checked with the Segha keymap round-trip (pure-Thaana QA).

  Start-Process .venv\Scripts\python.exe scripts\inference_server.py

API: POST /translate  {"text": "...", "num_beams": 4}  ->
     {"lines": [{"en", "dv", "roundtrip_ok"}], "seconds": float}
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "div-transliteration"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from keymap import keymap_to_thaana, thaana_to_keymap
from madlad_loader import load_madlad

PORT = 8734
ADAPTER = Path(__file__).resolve().parents[1] / "train" / "checkpoints" / "madlad3b-lora-r1"
PAPER_HTML = Path(__file__).resolve().parents[1] / "paper" / "paper.html"

print("loading model...", flush=True)
t0 = time.time()
TOKENIZER, MODEL = load_madlad()
from peft import PeftModel

MODEL = PeftModel.from_pretrained(MODEL, str(ADAPTER))
MODEL = MODEL.merge_and_unload()
MODEL = MODEL.to("cuda").eval()
GEN_LOCK = threading.Lock()
print(f"READY in {time.time() - t0:.0f}s", flush=True)


def translate(lines, num_beams=4):
    inputs = TOKENIZER(
        [f"<2dv> {l}" for l in lines],
        return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).to("cuda")
    with GEN_LOCK, torch.inference_mode():
        out = MODEL.generate(**inputs, max_new_tokens=256, num_beams=num_beams)
    return TOKENIZER.batch_decode(out, skip_special_tokens=True)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EN → DV translator (madlad3b-lora-r1)</title><style>
body{background:#111418;color:#e6e6e6;font-family:system-ui,sans-serif;
     max-width:780px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.1rem;color:#8ab4f8}
textarea{width:100%;min-height:110px;background:#1b1f26;color:#e6e6e6;
     border:1px solid #333;border-radius:8px;padding:.7rem;font-size:1rem;box-sizing:border-box}
button{background:#8ab4f8;color:#111;border:0;border-radius:8px;
     padding:.55rem 1.4rem;font-size:1rem;font-weight:600;cursor:pointer;margin-top:.5rem}
button:disabled{opacity:.5}
.card{background:#1b1f26;border:1px solid #2a2f37;border-radius:8px;
     padding:.8rem 1rem;margin-top:.8rem}
.en{color:#9aa0a6;font-size:.85rem;margin-bottom:.35rem}
.dv{direction:rtl;text-align:right;font-size:1.45rem;line-height:2.1}
.meta{color:#9aa0a6;font-size:.8rem;margin-top:.6rem}
.bad{color:#f28b82}.ok{color:#81c995}
</style></head><body>
<h1>English → Dhivehi &nbsp;·&nbsp; madlad400-3b-mt + LoRA r1 (chrF++ 61.8)
&nbsp;·&nbsp; <a href="/paper" style="color:#8ab4f8">paper</a></h1>
<textarea id="t" placeholder="Type English here. One sentence per line."></textarea><br>
<button id="go" onclick="run()">Translate</button>
<div id="out"></div>
<script>
async function run(){
  const btn=document.getElementById('go'), out=document.getElementById('out');
  const text=document.getElementById('t').value.trim();
  if(!text) return;
  btn.disabled=true; btn.textContent='Translating…';
  try{
    const r=await fetch('/translate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const d=await r.json();
    out.innerHTML=d.lines.map(l=>
      `<div class="card"><div class="en">${esc(l.en)}</div>`+
      `<div class="dv">${esc(l.dv)}</div>`+
      (l.roundtrip_ok?'':'<div class="meta bad">⚠ keymap round-trip failed (non-Thaana chars)</div>')+
      `</div>`).join('')+
      `<div class="meta">beam-4 · ${d.seconds.toFixed(1)}s</div>`;
  }catch(e){ out.innerHTML='<div class="card bad">'+esc(String(e))+'</div>'; }
  btn.disabled=false; btn.textContent='Translate';
}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
document.getElementById('t').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))run();});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, '{"status":"ready"}')
        elif self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/paper":
            self._send(200, PAPER_HTML.read_text(encoding="utf-8"),
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
            lines = [l.strip() for l in req["text"].splitlines() if l.strip()][:32]
            num_beams = min(int(req.get("num_beams", 4)), 8)
            t0 = time.time()
            preds = translate(lines, num_beams)
            result = {
                "lines": [
                    {
                        "en": en,
                        "dv": dv,
                        "roundtrip_ok": keymap_to_thaana(thaana_to_keymap(dv)) == dv,
                    }
                    for en, dv in zip(lines, preds)
                ],
                "seconds": time.time() - t0,
            }
            self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
