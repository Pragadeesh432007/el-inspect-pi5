"""EL solar-cell inspection dashboard (Raspberry Pi 5 or any PC; needs only numpy, pillow, onnxruntime).

    python pvel_app.py --demo --serve 8000          # cycle through bundled test cells, compare with ground truth
    python pvel_app.py --image cell.jpg             # one image or a folder of images
    python pvel_app.py --camera --serve 8000        # live Pi camera; dashboard at http://<pi>.local:8000
    --model pvel_fp32.onnx   compare with the FP32 model (default: pvel_int8.onnx)
    --smooth 3               average the probabilities over the last 3 frames (steadier live view)
    --interval 0             run as fast as possible (measures pipeline throughput)

Two pipelines (documented separately in the report):
  A  vision   : camera -> pre-process -> MobileNetV3-Small (ONNX Runtime, INT8, CPU) -> structured findings
  B  advisory : findings (JSON, no pixels) -> Llama 3.2 1B Instruct run by ExecuTorch (llama_main, CPU) -> advice
Rules decide pass/fail and severity; Llama explains the cause, impact and action. Everything runs locally.
    --llama-dir DIR          folder with llama_main, the .pte and tokenizer.model (default ~/executorch-deployment)
    --llama-wait             pause the inspection loop until Llama has written the advisory (good for demos)
    --no-llama               rule-based advice only

For every cell it produces a health report: status, severity, fault types with plain-language
meaning, recommended action, per-defect probabilities against their tuned thresholds, a frame
check, and timing. With --serve it shows a live dashboard (verdict, probabilities, performance,
session statistics, history) and offers a CSV export of the inspection log.

Limits: a standard Pi camera cannot capture real electroluminescence (EL needs a near-infrared
camera and a forward-biased cell). In --camera mode it photographs an EL image on a screen or print,
which exercises the full edge pipeline but is not a real EL capture station. The model classifies
the whole cell; it does not localise defects. Severity rules are our own engineering heuristics,
not an industry standard. Research prototype, not a certified inspection system.
"""
import argparse
import csv
import json
import os
import platform
import socket
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

try:
    from advisor import Advisor
except ImportError:                                           # advisor.py missing: dashboard still works
    Advisor = None

MEAN, STD, SIZE = 0.449, 0.226, 384
LEVELS = ["low", "moderate", "severe"]

# defect type -> (base severity, display name, plain-language meaning)
DEFECTS = {
    "black_core":    (2, "Black core", "dark inactive region in the cell (power loss)"),
    "short_circuit": (2, "Short circuit", "short-circuited / dead cell area"),
    "crack":         (1, "Crack", "micro-crack in the silicon"),
    "dislocation":   (1, "Dislocation", "crystal dislocation line"),
    "thick_line":    (1, "Thick line", "thick dark line (electrode / printing problem)"),
    "finger":        (0, "Finger defect", "broken or missing finger electrode"),
    "other":         (1, "Other", "other defect (corner, fragment, scratch, printing error)"),
}
ACTION = {
    -1: "No action needed. Cell looks healthy.",
    0: "Monitor. Re-inspect at the next scheduled check.",
    1: "Schedule a detailed inspection or repair.",
    2: "Remove from service. Repair or replace soon.",
}


# ----------------------------------------------------------------------------- model / pipeline
def load_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def to_cell(img):
    """PIL image -> 384x384 uint8 grayscale (centre square crop, same resize as training)"""
    g = img.convert("L")
    w, h = g.size
    s = min(w, h)
    g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(g.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8)


def infer(sess, cell, classes):
    x = ((cell.astype(np.float32) / 255.0 - MEAN) / STD)[None, None]
    t = time.perf_counter()
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    ms = (time.perf_counter() - t) * 1000
    p = 1 / (1 + np.exp(-logits))
    return {n: float(p[i]) for i, n in enumerate(classes + ["DEFECTIVE"])}, ms


def frame_check(cell):
    """Advisory checks on the 384x384 cell. Dark cells are NOT rejected: in PVEL-AD, fully dark cells are
    genuine short-circuit / dead cells, so darkness is only reported, not treated as an error."""
    c = cell.astype(np.float32)
    mean, std = float(c.mean()), float(c.std())
    lap = 4 * c[1:-1, 1:-1] - c[:-2, 1:-1] - c[2:, 1:-1] - c[1:-1, :-2] - c[1:-1, 2:]
    sharp = float(lap.var())
    bright = float((c > 240).mean())
    issues = []
    if std < 6:
        issues.append("Almost uniform frame: a dead cell, or no cell in view. Confirm the cell fills the frame.")
    elif mean < 20:
        issues.append("Very dark frame: dead cell or poor exposure. Confirm the cell is in view.")
    if mean > 200 or bright > 0.20:
        issues.append("Bright border or glare in the crop. Move closer so the cell fills the frame.")
    if std >= 20 and sharp < 40:
        issues.append("Image looks blurred. Refocus and hold the camera steady.")
    return {"mean": round(mean, 1), "contrast": round(std, 1), "sharpness": round(sharp, 1),
            "ok": not issues, "issues": issues}


def health_report(probs, thr, classes):
    """probs/thr over classes + 'DEFECTIVE' -> report dict"""
    found = [c for c in classes if probs[c] >= thr[c]]
    defective = probs["DEFECTIVE"] >= thr["DEFECTIVE"] or bool(found)
    if not defective:
        return {"status": "HEALTHY", "severity": "none", "faults": [], "action": ACTION[-1],
                "confidence": 1 - probs["DEFECTIVE"]}
    if found:
        lvl = max(DEFECTS[c][0] for c in found)
        if len(found) >= 3:                       # several different defects -> one level worse
            lvl = min(lvl + 1, 2)
        faults = sorted(found, key=lambda c: -probs[c])
    else:
        lvl, faults = 1, []                       # defective, type unclear
    return {"status": "DEFECTIVE", "severity": LEVELS[lvl], "faults": faults,
            "action": ACTION[lvl] + ("" if faults else " (defect type unclear: inspect manually)"),
            "confidence": probs["DEFECTIVE"]}


# ----------------------------------------------------------------------------- shared state
class State:
    paused = False
    source = "camera"
    notice = ""
    upload = None
    lock = threading.Lock()
    start = time.time()
    meta = {}
    current = None
    cell_png = b""
    history = deque(maxlen=14)
    thumbs = {}
    lat, pipe = deque(maxlen=200), deque(maxlen=200)
    total = healthy = defective = 0
    sev = {"low": 0, "moderate": 0, "severe": 0}
    types = {}
    truth_n = truth_ok = 0
    log_path = Path("inspection_log.csv")
    adv_log_path = Path("advisory_log.jsonl")
    advisory = None
    advisor = None


def png_bytes(arr_or_img, size=None):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    b = BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def sysinfo():
    out = {"cpu_temp_c": None, "ram_mb": None, "load1": None}
    try:
        out["cpu_temp_c"] = round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except Exception:
        pass
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS"):
                out["ram_mb"] = round(int(ln.split()[1]) / 1024)
    except Exception:
        pass
    try:
        out["load1"] = round(os.getloadavg()[0], 2)
    except Exception:
        pass
    return out


def pct(v, q):
    return round(float(np.percentile(list(v), q)), 1) if v else None


def snapshot():
    with State.lock:
        lat, pipe = list(State.lat), list(State.pipe)
        mean_pipe = float(np.mean(pipe)) if pipe else None
        return {
            "meta": State.meta, "uptime_s": int(time.time() - State.start), "now": time.time(), "paused": State.paused, "source": State.source, "notice": State.notice,
            "current": State.current, "history": list(State.history)[::-1],
            "advisory": State.advisory,
            "llama": State.advisor.stats() if State.advisor else {"enabled": False, "reason": "advisor module not loaded"},
            "perf": {"latency_ms": lat[-1] if lat else None,
                     "latency_mean": round(float(np.mean(lat)), 1) if lat else None,
                     "latency_p95": pct(lat, 95),
                     "pipeline_ms": round(mean_pipe, 1) if mean_pipe else None,
                     "pipeline_fps": round(1000 / mean_pipe, 1) if mean_pipe else None,
                     "frames": len(lat), **sysinfo()},
            "stats": {"total": State.total, "healthy": State.healthy, "defective": State.defective,
                      "yield_pct": round(100 * State.healthy / State.total, 1) if State.total else None,
                      "severity": dict(State.sev), "types": dict(State.types),
                      "truth_n": State.truth_n,
                      "truth_acc": round(100 * State.truth_ok / State.truth_n, 1) if State.truth_n else None},
        }


# ----------------------------------------------------------------------------- dashboard (single page)
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EL Inspect - Solar Cell Quality Assistant</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1c232c;--line:#2a323d;--tx:#e6edf3;--mut:#8b98a7;--blue:#58a6ff;
--none:#3fb950;--low:#d9b92e;--moderate:#f0883e;--severe:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:12px 22px;border-bottom:1px solid var(--line);background:#0b0e13;flex-wrap:wrap}
.logo{width:30px;height:30px;border-radius:7px;background:linear-gradient(135deg,#1f6feb,#3fb950);display:grid;place-items:center;font-weight:700}
h1{font-size:17px;margin:0;font-weight:650}.sub{color:var(--mut);font-size:12px}
.sp{flex:1}.chip{border:1px solid var(--line);background:var(--panel);padding:3px 10px;border-radius:99px;font-size:12px;color:var(--mut)}
.live{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--mut)}.dot{width:9px;height:9px;border-radius:50%;background:var(--none);box-shadow:0 0 8px var(--none)}
.dot.off{background:var(--severe);box-shadow:none}
main{padding:18px 22px;display:grid;gap:16px;grid-template-columns:repeat(12,1fr);max-width:1500px;margin:auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{margin:0 0 12px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600}
.c4{grid-column:span 4}.c5{grid-column:span 5}.c6{grid-column:span 6}.c7{grid-column:span 7}.c8{grid-column:span 8}.c12{grid-column:span 12}
@media(max-width:1000px){.c4,.c5,.c6,.c7,.c8{grid-column:span 12}}
.cellwrap{position:relative}.cellwrap img{width:100%;image-rendering:auto;border-radius:8px;border:1px solid var(--line);display:block;background:#000}
.tag{position:absolute;left:8px;top:8px;background:#000a;padding:2px 8px;border-radius:6px;font-size:12px}
.badge{display:inline-block;font-size:30px;font-weight:750;letter-spacing:.02em}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:650;font-size:12px;letter-spacing:.05em;margin-left:10px;vertical-align:middle;color:#0b0e13}
.meta{color:var(--mut);font-size:12px;margin:6px 0 14px}.meta b{color:var(--tx);font-weight:600}
.fault{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:9px 12px;margin-bottom:8px}
.fault .n{font-weight:650}.fault .d{color:var(--mut);font-size:12px}
.action{border-left:4px solid var(--blue);background:var(--panel2);padding:10px 12px;border-radius:6px;margin:12px 0}
.warn{border-left:4px solid var(--low);background:#2a2410;padding:8px 12px;border-radius:6px;margin:8px 0;font-size:12.5px}
.truth{font-size:12.5px;margin-top:8px}.ok{color:var(--none)}.bad{color:var(--severe)}
.bar{display:grid;grid-template-columns:110px 1fr 90px;align-items:center;gap:10px;margin:7px 0;font-size:13px}
.track{position:relative;height:12px;background:#0b0e13;border-radius:6px;border:1px solid var(--line)}
.fill{height:100%;border-radius:6px;background:#3b4a5c}.fill.hit{background:var(--severe)}
.tick{position:absolute;top:-3px;width:2px;height:16px;background:#e6edf3aa}
.val{color:var(--mut);font-variant-numeric:tabular-nums;text-align:right;font-size:12px}
.legend{color:var(--mut);font-size:11.5px;margin-top:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}.kpi .l{color:var(--mut);font-size:11.5px}
.stack{display:flex;height:14px;border-radius:7px;overflow:hidden;background:#0b0e13;border:1px solid var(--line);margin:8px 0}
table{width:100%;border-collapse:collapse;font-size:12.5px}th{color:var(--mut);text-align:left;font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:6px 8px;border-bottom:1px solid #20262f;vertical-align:middle}td img{width:34px;height:34px;border-radius:4px;display:block}
.sevtxt{font-weight:650}footer{padding:6px 22px 26px;color:var(--mut);font-size:12px;max-width:1500px;margin:auto}

.seg{display:flex;gap:6px}.seg button,.seg .up{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:12px}
.seg button.on{background:#1f6feb;border-color:#1f6feb}.seg .up{display:inline-block}.notice{background:#2a2410;border-bottom:1px solid #4d3f10;color:#f0d77a;padding:7px 22px;font-size:12.5px}
a.btn{color:var(--blue);text-decoration:none;border:1px solid var(--line);padding:4px 10px;border-radius:6px;font-size:12px;background:var(--panel)}
.empty{color:var(--mut);padding:30px;text-align:center}
.adv{display:grid;grid-template-columns:90px 1fr;gap:6px 12px;margin:8px 0}.adv .k{color:var(--blue);font-weight:650}
.src{display:inline-block;padding:2px 10px;border-radius:99px;font-size:11.5px;font-weight:650;color:#0b0e13;margin-left:8px;vertical-align:middle}
details{margin-top:10px}summary{cursor:pointer;color:var(--mut);font-size:12.5px}
pre{background:#0b0e13;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:12px;color:#c9d1d9;margin:8px 0 0}
</style></head><body>
<header><div class="logo">EL</div><div><h1>EL Inspect <span class="sub">Solar cell quality assistant</span></h1>
<div class="sub" id="devline"></div></div><div class="sp"></div>
<span class="chip" id="modelchip"></span><span class="chip" id="edgechip">CPU-only edge AI</span>
<div class="seg"><button id="b_camera" onclick="setSource('camera')">Camera</button><button id="b_demo" onclick="setSource('demo')">Demo cells</button>
<label class="up">Upload image<input type="file" id="fup" accept="image/*" onchange="uploadImg(this)" hidden></label>
<button id="pbtn" onclick="togglePause()">Pause</button></div>
<div class="live"><span class="dot" id="dot"></span><span id="liveTxt">connecting</span></div></header>
<div id="notice" class="notice" style="display:none"></div>
<main id="main"><div class="card c12 empty">Waiting for the first inspection...</div></main>
<footer>Research prototype, not a certified inspection system. The model classifies the whole cell and does not localise defects.
Severity levels are engineering rules, not an industry standard. In camera mode a standard Pi camera photographs an EL image on a screen or print;
real EL needs a near-infrared camera and a forward-biased cell. The advisory text is written by a small language model (Llama 3.2 1B, Built with Llama) and can be wrong:
severity and pass/fail come from rules and are never changed by it. &nbsp; <a class="btn" href="/export.csv">Download inspection log (CSV)</a>
<a class="btn" href="/report.json">Latest report (JSON)</a> <a class="btn" href="/advisory_log.jsonl">Advisory log (JSONL)</a></footer>
<script>
const COL={none:'var(--none)',low:'var(--low)',moderate:'var(--moderate)',severe:'var(--severe)'};
const SEVN={low:'Low',moderate:'Medium',severe:'High'};
const NAMES={crack:'Crack',finger:'Finger defect',black_core:'Black core',thick_line:'Thick line',short_circuit:'Short circuit',dislocation:'Dislocation',other:'Other',DEFECTIVE:'Overall defective'};
let lastId=null,lastOk=Date.now(),advOpen=false;
const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SRCCOL={llama:'var(--none)','llama (cached)':'var(--none)','llama (partly rule-based)':'var(--low)','rule-based':'var(--blue)'};
function advCard(s){
 const a=s.advisory,L=s.llama||{};
 const head=`<h2>AI advisory <span class="chip" style="text-transform:none;letter-spacing:0">${esc(L.model||'Llama')} · ${esc(L.runtime||'ExecuTorch')} · on-device</span></h2>`;
 if(!a)return `<div class="card c12">${head}<div class="legend">Waiting for the first inspection.${L.enabled===false&&L.reason?' '+esc(L.reason):''}</div></div>`;
 if(a.state=='none')return `<div class="card c12">${head}<div class="legend">${esc(a.note)} Llama is called for defective cells only.</div></div>`;
 const cur=s.current,stale=cur&&cur.id!==a.id?`<div class="legend" style="margin:0 0 6px">Current cell ${esc(cur.id)} is ${cur.status=='HEALTHY'?'healthy, so no new advisory is needed':'being processed'}. Showing the advisory for ${esc(a.id)}, the latest defective cell.</div>`:'';
 const ln=a.lines||{};
 const rows=['cause','impact','action'].map(k=>`<div class="k">${k[0].toUpperCase()+k.slice(1)}</div><div>${esc(ln[k])}</div>`).join('');
 const t=a.timing, kp=(v,l)=>`<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
 const timing=t?`<div class="kpis" style="margin-top:10px">${kp(f(t.load_s)+'<small> s</small>','Model load')}${kp(f(t.first_token_s,2)+'<small> s</small>','Time to first token')}${kp(f(t.tokens_per_s)+'','Tokens / s')}${kp(f(t.total_s)+'<small> s</small>','Total response')}${kp(L.peak_rss_mb==null?'-':L.peak_rss_mb+' MB','Llama peak RAM')}</div>`:'';
 const pend=a.state=='pending'?'<div class="warn">Llama is writing the advisory (about 10 s on the Pi). Preliminary rule-based text is shown.</div>':'';
 const g=(a.guard&&a.guard.length)?`<div class="warn">Guard: the ${a.guard.join(', ')} line from Llama contained a number that was not in the findings, so rule-based text was used for it.</div>`:'';
 return `<div class="card c12">${head}
  ${stale}<div class="meta"><b>${esc(a.id)}</b> · ${tm(a.time)} <span class="src" style="background:${SRCCOL[a.source]||'var(--mut)'}">${esc(a.source)}</span> &nbsp; ${esc(a.note)}</div>
  <div class="adv">${rows}</div>${pend}${g}${timing}
  <div class="legend">Llama receives the structured findings below (class, confidence, severity), not the image. Severity and pass/fail are decided by rules. Llama runs ${L.runs||0} time(s), ${L.cached||0} cached, ${L.fallbacks||0} fallback(s).</div>
  <details ${advOpen?'open':''} ontoggle="advOpen=this.open"><summary>Findings sent to Llama (JSON hand-off)</summary><pre>${esc(JSON.stringify(a.findings,null,1))}</pre></details></div>`}
const f=(v,d=1)=>v==null?'-':Number(v).toFixed(d);
const tm=t=>new Date(t*1000).toLocaleTimeString();
function bars(c){return Object.entries(c.probs).map(([k,o])=>`<div class="bar"><div>${NAMES[k]||k}</div>
<div class="track"><div class="fill ${o.hit?'hit':''}" style="width:${Math.min(100,o.p*100)}%"></div><div class="tick" style="left:${Math.min(100,o.thr*100)}%"></div></div>
<div class="val">${(o.p*100).toFixed(1)}% / ${(o.thr*100).toFixed(0)}%</div></div>`).join('')}
function render(s){
 document.getElementById('pbtn').textContent=s.paused?'Resume':'Pause';
 for(const k of ['camera','demo']){document.getElementById('b_'+k).className=(s.source==k&&!s.paused)?'on':''}
 const nt=document.getElementById('notice');nt.style.display=s.notice?'block':'none';nt.textContent=s.notice||'';
 document.getElementById('modelchip').textContent=s.meta.model+' · '+s.meta.precision+(s.paused?' · PAUSED':'');
 document.getElementById('devline').textContent=s.meta.device+' · '+s.meta.threads+' threads · input '+s.meta.input+' · uptime '+Math.floor(s.uptime_s/60)+'m '+(s.uptime_s%60)+'s';
 const c=s.current; if(!c)return;
 const col=COL[c.severity]||COL.none, st=s.stats, p=s.perf;
 const faults=c.faults.length?c.faults.map(x=>`<div class="fault"><div class="n">${x.name} <span style="float:right;color:var(--mut);font-weight:500">${(x.p*100).toFixed(1)}%</span></div><div class="d">${x.desc}</div></div>`).join(''):`<div class="fault"><div class="d">${c.status=='HEALTHY'?'No defect type above its threshold.':'Defective, but no single type above its threshold. Inspect manually.'}</div></div>`;
 const warns=c.quality.issues.map(i=>`<div class="warn">Frame check: ${i}</div>`).join('');
 let truth='';if(c.truth!=null){truth=`<div class="truth">Ground truth: <b>${c.truth.length?c.truth.join(', '):'good cell'}</b> &nbsp; <span class="${c.match?'ok':'bad'}">${c.match?'prediction agrees':'prediction differs'}</span></div>`}
 const tot=Math.max(1,st.defective);const sv=['severe','moderate','low'].map(k=>`<div style="width:${100*st.severity[k]/tot}%;background:${COL[k]}" title="${SEVN[k]}: ${st.severity[k]}"></div>`).join('');
 const tr=Object.entries(st.types).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<div class="bar" style="grid-template-columns:110px 1fr 40px"><div>${NAMES[k]}</div><div class="track"><div class="fill" style="width:${100*v/Math.max(1,st.defective)}%;background:var(--blue)"></div></div><div class="val">${v}</div></div>`).join('')||'<div class="legend">No defects yet.</div>';
 const rows=s.history.map(h=>`<tr><td><img src="/thumb/${h.id}.png"></td><td>${h.id}</td><td>${tm(h.time)}</td><td class="sevtxt" style="color:${h.status=='HEALTHY'?COL.none:COL[h.severity]}">${h.status}</td><td>${h.severity=='none'?'-':SEVN[h.severity]}</td><td>${h.faults.join(', ')||'-'}</td><td>${(h.p_def*100).toFixed(0)}%</td><td>${f(h.latency_ms)} ms</td></tr>`).join('');
 document.getElementById('main').innerHTML=`
 <div class="card c4"><h2>Inspected cell</h2><div class="cellwrap"><img id="cellimg" src="/cell.png?id=${c.id}"><span class="tag">${c.source}</span></div>
  <div class="legend">${c.note||''}</div></div>
 <div class="card c8"><h2>Health report</h2>
  <span class="badge" style="color:${c.status=='HEALTHY'?COL.none:col}">${c.status}</span>${c.severity!='none'?`<span class="pill" style="background:${col}">${SEVN[c.severity].toUpperCase()} SEVERITY</span>`:''}
  <div class="meta"><b>${c.id}</b> · ${tm(c.time)} · source <b>${c.source}</b> · P(defective) <b>${(c.p_def*100).toFixed(1)}%</b> (threshold ${(c.thr_def*100).toFixed(0)}%)</div>
  <h2 style="margin-top:6px">Detected faults</h2>${faults}
  <div class="action"><b>Recommended action</b><br>${c.action}</div>${warns}${truth}</div>
 ${advCard(s)}
 <div class="card c7"><h2>Defect probabilities vs decision thresholds</h2>${bars(c)}
  <div class="legend">Bar = model probability, white tick = tuned decision threshold, red = at or above threshold. Thresholds were tuned on validation data only.</div></div>
 <div class="card c5"><h2>Edge performance</h2><div class="kpis">
  <div class="kpi"><div class="v">${f(p.latency_ms)}<small> ms</small></div><div class="l">Inference (last)</div></div>
  <div class="kpi"><div class="v">${f(p.latency_mean)}<small> ms</small></div><div class="l">Inference (mean)</div></div>
  <div class="kpi"><div class="v">${f(p.latency_p95)}<small> ms</small></div><div class="l">Inference (p95)</div></div>
  <div class="kpi"><div class="v">${f(p.pipeline_ms)}<small> ms</small></div><div class="l">End-to-end</div></div>
  <div class="kpi"><div class="v">${f(p.pipeline_fps)}</div><div class="l">Pipeline FPS</div></div>
  <div class="kpi"><div class="v">${p.cpu_temp_c==null?'-':f(p.cpu_temp_c)+'°C'}</div><div class="l">CPU temperature</div></div>
  <div class="kpi"><div class="v">${p.ram_mb==null?'-':p.ram_mb+' MB'}</div><div class="l">App memory</div></div>
  <div class="kpi"><div class="v">${p.load1==null?'-':p.load1}</div><div class="l">CPU load (1 min)</div></div></div>
  <div class="legend">Frame check: brightness ${c.quality.mean}, contrast ${c.quality.contrast}, sharpness ${c.quality.sharpness}</div></div>
 <div class="card c5"><h2>Session statistics</h2><div class="kpis">
  <div class="kpi"><div class="v">${st.total}</div><div class="l">Cells inspected</div></div>
  <div class="kpi"><div class="v" style="color:var(--none)">${st.healthy}</div><div class="l">Healthy</div></div>
  <div class="kpi"><div class="v" style="color:var(--severe)">${st.defective}</div><div class="l">Defective</div></div>
  <div class="kpi"><div class="v">${st.yield_pct==null?'-':st.yield_pct+'%'}</div><div class="l">Pass rate</div></div>
  ${st.truth_acc!=null?`<div class="kpi"><div class="v">${st.truth_acc}%</div><div class="l">Agrees with truth (${st.truth_n})</div></div>`:''}</div>
  <h2 style="margin-top:14px">Severity of defective cells</h2><div class="stack">${sv}</div>
  <div class="legend"><span style="color:var(--severe)">■</span> High ${st.severity.severe} &nbsp; <span style="color:var(--moderate)">■</span> Medium ${st.severity.moderate} &nbsp; <span style="color:var(--low)">■</span> Low ${st.severity.low}</div>
  <h2 style="margin-top:14px">Defect types found</h2>${tr}</div>
 <div class="card c7"><h2>Inspection history</h2><table><tr><th></th><th>ID</th><th>Time</th><th>Status</th><th>Severity</th><th>Faults</th><th>P(def)</th><th>Latency</th></tr>${rows}</table></div>`;
}
async function setSource(x){try{await fetch('/api/source?s='+x,{method:'POST'});tick()}catch(e){}}
async function uploadImg(el){const f=el.files[0];if(!f)return;try{await fetch('/api/upload',{method:'POST',body:f});el.value='';setTimeout(tick,600)}catch(e){}}
async function togglePause(){try{await fetch('/api/pause',{method:'POST'});tick()}catch(e){}}
async function tick(){try{const r=await fetch('/api/state');const s=await r.json();lastOk=Date.now();
 if(s.current&&s.current.id!==lastId){lastId=s.current.id;}render(s);
 document.getElementById('liveTxt').textContent='live · updated '+tm(s.now);document.getElementById('dot').className='dot'}
 catch(e){document.getElementById('liveTxt').textContent='disconnected';document.getElementById('dot').className='dot off'}}
setInterval(tick,1000);tick();
</script></body></html>"""


def serve(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, extra=None):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path, _, q = self.path.partition("?")
            if path == "/api/pause":
                State.paused = not State.paused
                if not State.paused:
                    State.notice = ""
                self.send(json.dumps({"paused": State.paused}).encode(), "application/json")
            elif path == "/api/source":
                src = q.replace("s=", "")
                if src in ("camera", "demo"):
                    State.source, State.paused, State.notice = src, False, ""
                self.send(json.dumps({"source": State.source}).encode(), "application/json")
            elif path == "/api/upload":
                n = int(self.headers.get("Content-Length", 0))
                if 0 < n < 20_000_000:
                    State.upload = self.rfile.read(n)
                self.send(b'{"ok": true}', "application/json")
            else:
                self.send_error(404)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                self.send(json.dumps(snapshot()).encode(), "application/json")
            elif path == "/cell.png":
                with State.lock:
                    self.send(State.cell_png, "image/png")
            elif path.startswith("/thumb/"):
                with State.lock:
                    b = State.thumbs.get(path[7:-4], b"")
                self.send(b, "image/png")
            elif path == "/export.csv":
                b = State.log_path.read_bytes() if State.log_path.exists() else b""
                self.send(b, "text/csv", {"Content-Disposition": "attachment; filename=inspection_log.csv"})
            elif path == "/advisory_log.jsonl":
                b = State.adv_log_path.read_bytes() if State.adv_log_path.exists() else b""
                self.send(b, "application/x-ndjson", {"Content-Disposition": "attachment; filename=advisory_log.jsonl"})
            elif path == "/report.json":
                with State.lock:
                    self.send(json.dumps(State.current, indent=2).encode(), "application/json")
            else:
                self.send(PAGE.encode(), "text/html; charset=utf-8")

    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start()
    print(f"dashboard: http://{socket.gethostname()}.local:{port}   (or http://<this-device-ip>:{port})")


# ----------------------------------------------------------------------------- report image (for figures)
def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render(cell, rec):
    W, H, P = 384, 384, 14
    img = Image.new("RGB", (W + 470, H + 2 * P), (24, 26, 30))
    img.paste(Image.fromarray(cell).convert("RGB"), (P, P))
    d = ImageDraw.Draw(img)
    colour = {"none": (80, 200, 120), "low": (230, 200, 70), "moderate": (240, 150, 50),
              "severe": (235, 70, 70)}[rec["severity"]]
    x, y = W + 2 * P, P
    d.text((x, y), rec["status"], fill=colour, font=font(30)); y += 42
    d.text((x, y), f"Severity: {({'low':'LOW','moderate':'MEDIUM','severe':'HIGH'}).get(rec['severity'], rec['severity'].upper())}", fill=colour, font=font(22)); y += 34
    names = [f["name"] for f in rec["faults"]]
    d.text((x, y), "Faults: " + (", ".join(names) if names else "none"), fill=(235, 235, 235), font=font(18)); y += 28
    for f_ in rec["faults"][:3]:
        d.text((x, y), f" - {f_['name']}: {f_['desc']}", fill=(180, 180, 190), font=font(13)); y += 19
    y += 6
    line = ""
    for w in ("Action: " + rec["action"]).split():
        if len(line + w) > 48:
            d.text((x, y), line, fill=(235, 235, 235), font=font(16)); y += 22; line = ""
        line += w + " "
    d.text((x, y), line, fill=(235, 235, 235), font=font(16)); y += 30
    d.text((x, y), f"P(defective) {rec['p_def']:.2f}   latency {rec['latency_ms']:.1f} ms",
           fill=(150, 200, 255), font=font(15)); y += 22
    d.text((x, y), f"{rec['id']}  model: {rec['model']}", fill=(120, 120, 130), font=font(13))
    return img


# ----------------------------------------------------------------------------- main
def start_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
    cam.start()
    try:
        cam.set_controls({"AfMode": 2})               # continuous autofocus (Camera Module 3)
    except Exception:
        pass
    time.sleep(1.5)
    return cam


def run_app(handle, d, classes, interval):
    """Application mode: everything is controlled from the dashboard (camera / demo / upload / pause)."""
    X = Y = order = None
    cam, i = None, 0
    while True:
        up = State.upload
        if up is not None:                                  # an image was uploaded from the dashboard
            State.upload = None
            try:
                t = time.perf_counter()
                cell = to_cell(Image.open(BytesIO(up)))
                State.paused = True                         # keep the uploaded result on screen
                State.notice = "Showing the uploaded image. Press Resume to go back to live inspection."
                handle(cell, "Uploaded image", "Image uploaded from the dashboard", None, (time.perf_counter() - t) * 1000)
            except Exception as e:
                State.notice = f"Could not read that image ({str(e)[:60]})."
            continue
        if State.paused:
            time.sleep(0.15)
            continue
        if State.source == "camera":
            if cam is None:
                try:
                    cam = start_camera()
                except Exception as e:
                    State.notice = f"Camera not available ({str(e)[:70]}). Showing demo cells instead."
                    State.source = "demo"
                    continue
            t0 = time.perf_counter()
            frame = cam.capture_array()[:, :, ::-1]
            cell = to_cell(Image.fromarray(frame))
            handle(cell, "Pi camera", "Live camera frame (centre square crop)", None, (time.perf_counter() - t0) * 1000)
        else:
            if X is None:
                try:
                    X, Y = np.load(d / "sample_x.npy"), np.load(d / "sample_y.npy")
                    order = np.random.RandomState(0).permutation(len(X))
                except Exception as e:
                    State.notice = f"Demo cells not found ({str(e)[:60]}). Use Camera or Upload."
                    State.paused = True
                    continue
            k = int(order[i % len(order)])
            i += 1
            truth = [DEFECTS[c][1] for j, c in enumerate(classes) if Y[k][j]]
            handle(X[k], f"sample {k}", "Bundled test cell (not used for training)", truth)
        end = time.time() + interval
        while time.time() < end and State.upload is None and not State.paused:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder with the models, thresholds.json, classes.json")
    ap.add_argument("--model", default="pvel_int8.onnx")
    ap.add_argument("--threads", type=int, default=4)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="image file or folder")
    g.add_argument("--demo", action="store_true", help="cycle through sample_x.npy")
    g.add_argument("--camera", action="store_true", help="live Pi camera")
    g.add_argument("--app", action="store_true", help="application mode: dashboard buttons choose camera / demo / upload")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between camera/demo frames")
    ap.add_argument("--serve", type=int, default=0, help="port for the browser dashboard")
    ap.add_argument("--n", type=int, default=0, help="stop after N frames (0 = run until Ctrl+C)")
    ap.add_argument("--smooth", type=int, default=1, help="average probabilities over the last N frames")
    ap.add_argument("--save-report", action="store_true", help="write report.png / report.json for every frame")
    ap.add_argument("--llama-dir", default="~/executorch-deployment",
                    help="folder with llama_main, libllama_runner.so, the .pte and tokenizer.model")
    ap.add_argument("--llama-model", default="Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte")
    ap.add_argument("--llama-tokenizer", default="tokenizer.model")
    ap.add_argument("--llama-timeout", type=int, default=120, help="seconds before falling back to rule-based advice")
    ap.add_argument("--llama-wait", action="store_true",
                    help="wait for Llama's advisory before inspecting the next cell (good for demos)")
    ap.add_argument("--llama-temperature", type=float, default=0.0,
                    help="llama_main sampling temperature; 0 = deterministic (same findings, same advice)")
    ap.add_argument("--no-llama", action="store_true", help="rule-based advice only")
    a = ap.parse_args()

    d = Path(a.dir)
    classes = json.loads((d / "classes.json").read_text())
    thr_file = "thresholds_int8.json" if ("int8" in a.model and (d / "thresholds_int8.json").exists()) \
        else "thresholds.json"                     # the INT8 model has its own validation-tuned thresholds
    thr = json.loads((d / thr_file).read_text())
    sess = load_session(d / a.model, a.threads)
    for _ in range(3):                                    # warm-up
        infer(sess, np.zeros((SIZE, SIZE), np.uint8), classes)
    State.meta = {"model": a.model, "precision": "INT8" if "int8" in a.model else "FP32",
                  "threads": a.threads, "input": f"{SIZE}x{SIZE} gray", "thresholds": thr_file,
                  "device": f"{platform.node()} · {platform.machine()}"}
    print(f"model {a.model} | thresholds {thr_file} | {State.meta['device']}")
    new_log = not State.log_path.exists()
    logf = open(State.log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["id", "time", "source", "status", "severity", "faults", "p_defective", "latency_ms",
                       "pipeline_ms"] + classes)
    def on_advisory(adv):
        with State.lock:
            if adv["state"] == "none" and State.advisory is not None:
                return                                    # keep showing the last real advisory for a healthy cell
            State.advisory = adv

    if Advisor is not None:
        State.advisor = Advisor(a.llama_dir, a.llama_model, a.llama_tokenizer, on_advisory,
                                log_path=State.adv_log_path, timeout=a.llama_timeout, enabled=not a.no_llama,
                                temperature=a.llama_temperature)
        print("advisor: Llama via ExecuTorch ready" if State.advisor.available
              else f"advisor: {State.advisor.reason}")
    else:
        print("advisor: advisor.py not found next to pvel_app.py; rule-based advice only")
    if a.serve:
        serve(a.serve)
    recent = deque(maxlen=max(1, a.smooth))
    counter = [0]

    def handle(cell, source, note="", truth=None, pre_ms=0.0):
        probs, ms = infer(sess, cell, classes)
        recent.append(probs)
        if len(recent) > 1:                                # temporal smoothing
            probs = {k: float(np.mean([r[k] for r in recent])) for k in probs}
        rep = health_report(probs, thr, classes)
        counter[0] += 1
        rid, now = f"INS-{counter[0]:04d}", time.time()
        faults = [{"key": c, "name": DEFECTS[c][1], "desc": DEFECTS[c][2], "p": round(probs[c], 4)}
                  for c in rep["faults"]]
        rec = {"id": rid, "time": now, "source": source, "note": note, "status": rep["status"],
               "severity": rep["severity"], "faults": faults, "action": rep["action"],
               "p_def": round(probs["DEFECTIVE"], 4), "thr_def": thr["DEFECTIVE"],
               "probs": {k: {"p": round(probs[k], 4), "thr": thr[k], "hit": probs[k] >= thr[k]}
                         for k in classes + ["DEFECTIVE"]},
               "quality": frame_check(cell), "latency_ms": round(ms, 2),
               "pipeline_ms": round(pre_ms + ms, 2), "model": a.model,
               "truth": truth, "match": None}
        if truth is not None:
            rec["match"] = (rep["status"] == "DEFECTIVE") == bool(truth)
        with State.lock:
            State.current, State.cell_png = rec, png_bytes(cell)
            State.thumbs[rid] = png_bytes(cell, 68)
            State.history.append({k: rec[k] for k in ("id", "time", "status", "severity", "p_def", "latency_ms")}
                                 | {"faults": [f_["name"] for f_ in faults]})
            for old in [k for k in State.thumbs if k not in {h["id"] for h in State.history}]:
                del State.thumbs[old]
            State.lat.append(ms); State.pipe.append(pre_ms + ms)
            State.total += 1
            if rep["status"] == "HEALTHY":
                State.healthy += 1
            else:
                State.defective += 1
                State.sev[rep["severity"]] += 1
                for c in rep["faults"]:
                    State.types[c] = State.types.get(c, 0) + 1
            if rec["match"] is not None:
                State.truth_n += 1; State.truth_ok += int(rec["match"])
        logw.writerow([rid, datetime.fromtimestamp(now).isoformat(timespec="seconds"), source, rep["status"],
                       rep["severity"], ";".join(rep["faults"]), rec["p_def"], rec["latency_ms"],
                       rec["pipeline_ms"]] + [rec["probs"][c]["p"] for c in classes]); logf.flush()
        chk = "" if rec["quality"]["ok"] else " | frame: " + rec["quality"]["issues"][0]
        print(f"[{rid} {source}] {rep['status']} | {rep['severity']} | {', '.join(rep['faults']) or '-'} | "
              f"P(def)={probs['DEFECTIVE']:.2f} | {ms:.1f} ms{chk}")
        if State.advisor is not None:                      # hand the findings to the Llama pipeline
            State.advisor.submit(rec)
            if (a.llama_wait or a.image) and State.advisor.available and rep["status"] == "DEFECTIVE":
                State.advisor.wait_idle(a.llama_timeout + 10)
        if a.save_report or a.image:
            rec["advisory"] = State.advisory
            render(cell, rec).save("report.png")
            Path("report.json").write_text(json.dumps(rec, indent=2))

    n = 0
    if a.app:
        if not a.serve:
            a.serve = 8000
            serve(8000)
        run_app(handle, d, classes, a.interval)
    elif a.demo:
        X = np.load(d / "sample_x.npy"); Y = np.load(d / "sample_y.npy")
        order = np.random.RandomState(0).permutation(len(X))
        while True:
            for i in order:
                while State.paused:
                    time.sleep(0.2)
                truth = [DEFECTS[c][1] for k, c in enumerate(classes) if Y[i][k]]
                handle(X[i], f"sample {i}", "Bundled test cell (not used for training)", truth)
                n += 1
                if a.n and n >= a.n:
                    return
                time.sleep(a.interval)
    elif a.image:
        p = Path(a.image)
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) \
            if p.is_dir() else [p]
        for f in files:
            t = time.perf_counter()
            cell = to_cell(Image.open(f))
            handle(cell, f.name, "Image file", None, (time.perf_counter() - t) * 1000)
        if a.serve:
            print("serving the last result; Ctrl+C to stop"); time.sleep(10 ** 6)
    else:
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise SystemExit("picamera2 not found. Install: sudo apt install python3-picamera2 "
                             "(venv must be created with --system-site-packages)")
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        cam.start()
        try:
            cam.set_controls({"AfMode": 2})               # continuous autofocus (Camera Module 3)
        except Exception:
            pass
        time.sleep(1.5)
        print("camera running; fill the frame with the EL cell. Ctrl+C to stop.")
        while True:
            while State.paused:
                time.sleep(0.2)
            t0 = time.perf_counter()
            frame = cam.capture_array()[:, :, ::-1]       # BGR -> RGB
            cell = to_cell(Image.fromarray(frame))
            handle(cell, "Pi camera", "Live camera frame (centre square crop)", None,
                   (time.perf_counter() - t0) * 1000)
            n += 1
            if a.n and n >= a.n:
                break
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
