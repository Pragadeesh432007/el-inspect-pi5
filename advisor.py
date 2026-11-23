"""Llama advisory component (pipeline B). Runs on-device through ExecuTorch.

The vision model (pipeline A, ONNX Runtime) turns an image into structured findings. This module turns those
findings (never pixels) into a short operator advisory with a Llama 3.2 model executed by ExecuTorch's
`llama_main` runner on the Raspberry Pi CPU.

Division of labour:
  * Rules decide pass/fail and severity (pvel_app.health_report). Llama never changes them.
  * Llama explains the likely cause, the impact and the action, in three short lines.
  * If Llama is missing, too slow, or answers badly, the rule-based text is shown instead and the dashboard says so.
  * A guard rejects any line that contains a number that was not in the findings (small models invent figures).

Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License.
"""
import json
import os
import re
import resource
import subprocess
import threading
import time
from pathlib import Path

SEV_NAME = {"low": "Low", "moderate": "Medium", "severe": "High"}

# Hedged, generic engineering text. Used when Llama is unavailable, and to replace any Llama line that fails the guard.
CAUSE = {
    "crack": "Likely mechanical stress during handling, lamination or transport.",
    "finger": "Likely a printing or firing fault in the finger electrodes.",
    "black_core": "Likely a poor wafer region or cell-processing fault that leaves an inactive area.",
    "thick_line": "Likely a screen-printing or electrode-line fault.",
    "short_circuit": "Likely a shunted or short-circuited region, or a fully inactive cell.",
    "dislocation": "Likely crystal-growth defects in the silicon wafer.",
    "other": "Defect type is unspecific; possible handling damage or printing error.",
}
IMPACT = {
    "high": "Significant local power loss and possible hot-spot risk.",
    "medium": "Some power loss, and the defect may worsen under thermal cycling.",
    "low": "Small effect on output now; worth monitoring over time.",
}

SYSTEM = (
    "You are a quality-control assistant for solar cell electroluminescence (EL) inspection. "
    "A vision model classified one cell and gave its findings as JSON. Severity and rule_action are fixed by rules: "
    "do not change them. Reply with exactly three lines, each starting with its label, and nothing else:\n"
    "Cause: <one sentence based on typical_cause>\nImpact: <one sentence>\nAction: <one sentence that follows rule_action>\n"
    "Each sentence has 10 to 25 words, hedged where unsure (likely, possibly), and uses only the findings. "
    "If no defect type is given, say the cause cannot be judged from the image and recommend manual inspection. "
    "Do not write any numbers, percentages or markdown."
)
EXAMPLE_IN = ('Findings: {"result":"DEFECTIVE","severity":"Medium","defects":[{"type":"Finger defect",'
              '"meaning":"broken or missing finger electrode","typical_cause":"Likely a printing or firing fault in the '
              'finger electrodes."}],"rule_action":"Schedule a detailed inspection or repair."}'
              "\nWrite the three lines.")
EXAMPLE_OUT = ("Cause: The finger electrodes were likely printed or fired imperfectly, leaving a broken line.\n"
               "Impact: Current collection drops near the break, so the cell delivers somewhat less power.\n"
               "Action: Schedule a detailed inspection and repair or replace the cell if the break is confirmed.")

MARK = "<|start_header_id|>assistant<|end_header_id|>"
NUM = re.compile(r"\d+(?:[.,]\d+)?")
KEYS = ("cause", "impact", "action")


# ----------------------------------------------------------------------------- hand-off + prompt
def build_findings(rec):
    """The JSON hand-off from the vision pipeline to Llama (structured findings, no pixels)."""
    f = {
        "inspection_id": rec["id"],
        "result": rec["status"],
        "severity": SEV_NAME.get(rec["severity"], "None"),
        "overall_confidence_pct": round(rec["p_def"] * 100),
        "scope": "whole-cell classification, defect position unknown",
        "defects": [{"type": x["name"], "meaning": x["desc"], "confidence_pct": round(x["p"] * 100),
                     "typical_cause": CAUSE.get(x["key"], "")} for x in rec["faults"]],
        "rule_action": rec["action"],
    }
    if not rec["faults"]:
        f["note"] = "defective but no defect type identified"
    return f


def build_prompt(findings):
    user = "Findings: " + json.dumps(findings, separators=(",", ":")) + "\nWrite the three lines."
    turn = lambda role, text: f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
    prompt = (turn("system", SYSTEM) + turn("user", EXAMPLE_IN) + turn("assistant", EXAMPLE_OUT)
              + turn("user", user) + MARK + "\n\n")
    return prompt, user


def rule_lines(rec):
    keys = [x["key"] for x in rec["faults"]]
    cause = " ".join(CAUSE[k] for k in keys[:2]) if keys else \
        "Defect type is unclear, so the cause cannot be judged from the image."
    sev = SEV_NAME.get(rec["severity"], "Low").lower()
    return {"cause": cause, "impact": IMPACT.get(sev, IMPACT["low"]), "action": rec["action"]}


# ----------------------------------------------------------------------------- parsing llama_main output
def split_answer(out):
    i = out.rfind(MARK)
    txt = out[i + len(MARK):] if i >= 0 else out
    txt = txt.split("<|eot_id|>")[0]
    txt = re.sub(r"PyTorchObserver.*", "", txt, flags=re.S)
    return txt.strip()


def extract_lines(txt):
    t = txt.replace("**", "").replace("`", "")
    found = {}
    for m in re.finditer(r"(?im)^\s*(?:[-*•]\s*|\d+[.)]\s*)?(cause|impact|action)\s*[:\-]\s*(.+?)\s*$", t):
        k = m.group(1).lower()
        if k not in found:
            found[k] = m.group(2).strip()
    if not found:                                          # three plain lines without labels: map them in order
        plain = [l.strip(" -*\u2022") for l in t.splitlines() if l.strip() and not l.strip().endswith(":")]
        if len(plain) == 3:
            found = dict(zip(KEYS, plain))
    return found


def clip(s, limit=170):
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "..."


def parse_observer(*texts):
    for t in texts:
        m = re.search(r"PyTorchObserver\s+(\{.*?\})", t or "", flags=re.S)
        if m:
            try:
                o = json.loads(m.group(1))
            except ValueError:
                continue
            inf = (o["inference_end_ms"] - o["inference_start_ms"]) / 1000.0
            gen = o.get("generated_tokens", 0)
            return {"prompt_tokens": o.get("prompt_tokens"), "generated_tokens": gen,
                    "load_s": round((o["model_load_end_ms"] - o["model_load_start_ms"]) / 1000.0, 2),
                    "first_token_s": round((o["first_token_ms"] - o["inference_start_ms"]) / 1000.0, 2),
                    "generate_s": round(inf, 2),
                    "tokens_per_s": round(gen / inf, 1) if inf > 0 and gen else None,
                    "total_s": round((o["inference_end_ms"] - o["model_load_start_ms"]) / 1000.0, 2)}
    return None


# ----------------------------------------------------------------------------- the advisor
class Advisor:
    def __init__(self, llama_dir, model, tokenizer, on_update, log_path="advisory_log.jsonl",
                 seq_len=512, timeout=120, enabled=True, runner=None, temperature=None):
        self.dir = Path(llama_dir).expanduser()
        self.exe, self.model, self.tok = self.dir / "llama_main", self.dir / model, self.dir / tokenizer
        self.model_name = Path(model).stem
        self.on_update, self.log_path = on_update, Path(log_path)
        self.seq_len, self.timeout, self.runner, self.temperature = seq_len, timeout, runner, temperature
        self.available, self.reason = False, ""
        if not enabled:
            self.reason = "Llama advisor switched off (--no-llama). Rule-based advice is shown."
        elif runner is None and not all(p.exists() for p in (self.exe, self.model, self.tok)):
            missing = [p.name for p in (self.exe, self.model, self.tok) if not p.exists()]
            self.reason = f"Llama files not found in {self.dir} ({', '.join(missing)}). Rule-based advice is shown."
        else:
            self.available = True
        self._cv = threading.Condition()
        self._pending, self._busy = None, False
        self._cache = {}
        self._lock = threading.Lock()
        self._st = {"runs": 0, "cached": 0, "fallbacks": 0, "guard_hits": 0, "superseded": 0,
                    "load_s": [], "first_token_s": [], "tokens_per_s": [], "total_s": [], "peak_rss_mb": None,
                    "last_error": ""}
        if self.available:
            threading.Thread(target=self._worker, daemon=True).start()

    # -- public
    def stats(self):
        with self._lock:
            s = self._st
            mean = lambda v: round(sum(v) / len(v), 2) if v else None
            return {"enabled": self.available, "reason": self.reason, "model": self.model_name,
                    "runtime": "ExecuTorch (llama_main), CPU", "busy": self._busy or self._pending is not None,
                    "runs": s["runs"], "cached": s["cached"], "fallbacks": s["fallbacks"],
                    "guard_hits": s["guard_hits"], "superseded": s["superseded"],
                    "mean_load_s": mean(s["load_s"]), "mean_first_token_s": mean(s["first_token_s"]),
                    "mean_tokens_per_s": mean(s["tokens_per_s"]), "mean_total_s": mean(s["total_s"]),
                    "peak_rss_mb": s["peak_rss_mb"], "last_error": s["last_error"]}

    def submit(self, rec):
        """Called for every inspection. Healthy cells need no advisory; defective cells get one."""
        if rec["status"] == "HEALTHY":
            self.on_update({"id": rec["id"], "state": "none", "source": "-", "lines": None, "findings": None,
                            "timing": None, "guard": [], "time": time.time(),
                            "note": "No defect found, so no advisory is needed."})
            return
        findings, rules = build_findings(rec), rule_lines(rec)
        if not self.available:
            self._final(rec["id"], "rule", "rule-based", rules, findings, None, [], self.reason)
            return
        key = (rec["severity"], tuple(x["key"] for x in rec["faults"]), bool(findings.get("frame_issues")))
        with self._lock:
            hit = self._cache.get(key)
        if hit:
            with self._lock:
                self._st["cached"] += 1
            self._final(rec["id"], "ready", "llama (cached)", hit["lines"], findings, hit["timing"], hit["guard"],
                        f"Same findings as {hit['id']}; its advisory is reused.")
            return
        self.on_update({"id": rec["id"], "state": "pending", "source": "rule-based", "lines": rules,
                        "findings": findings, "timing": None, "guard": [], "time": time.time(),
                        "note": "Llama is writing the advisory. Preliminary rule-based text is shown meanwhile."})
        with self._cv:
            if self._pending is not None:
                with self._lock:
                    self._st["superseded"] += 1
            self._pending = (rec, findings, rules, key)
            self._cv.notify_all()

    def wait_idle(self, timeout=120):
        end = time.time() + timeout
        with self._cv:
            while (self._busy or self._pending is not None) and time.time() < end:
                self._cv.wait(0.3)

    # -- internals
    def _final(self, rid, state, source, lines, findings, timing, guard, note, raw=None):
        adv = {"id": rid, "state": state, "source": source, "lines": lines, "findings": findings,
               "timing": timing, "guard": guard, "note": note, "time": time.time()}
        if raw:
            adv["raw"] = raw[:600]                        # what Llama actually said, kept for debugging
        self.on_update(adv)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(adv, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _worker(self):
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                job, self._pending, self._busy = self._pending, None, True
            try:
                self._do(*job)
            except Exception as e:                       # never let the advisor take the app down
                with self._lock:
                    self._st["last_error"] = f"{type(e).__name__}: {e}"
                    self._st["fallbacks"] += 1
                self._final(job[0]["id"], "rule", "rule-based", job[2], job[1], None, [],
                            f"Llama failed ({type(e).__name__}); rule-based advice is shown.")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _run_llama(self, prompt):
        if self.runner:
            return self.runner(prompt)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{self.dir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [str(self.exe), "--model_path", str(self.model), "--tokenizer_path", str(self.tok),
               "--seq_len", str(self.seq_len)]
        if self.temperature is not None:
            cmd += ["--temperature", str(self.temperature)]
        cmd += ["--prompt", prompt]
        p = subprocess.run(cmd, cwd=str(self.dir), env=env, capture_output=True, text=True,
                           errors="replace", timeout=self.timeout)
        return p.stdout, p.stderr, p.returncode

    def _do(self, rec, findings, rules, key):
        prompt, user = build_prompt(findings)
        t0 = time.perf_counter()
        try:
            out, err, rc = self._run_llama(prompt)
        except subprocess.TimeoutExpired:
            with self._lock:
                self._st["last_error"], self._st["fallbacks"] = f"timeout after {self.timeout}s", self._st["fallbacks"] + 1
            self._final(rec["id"], "rule", "rule-based", rules, findings, None, [],
                        f"Llama took longer than {self.timeout} s; rule-based advice is shown.")
            return
        wall = time.perf_counter() - t0
        timing = parse_observer(out, err)
        if timing:
            timing["wall_s"] = round(wall, 2)
        got = extract_lines(split_answer(out))
        if rc != 0 or len(got) < 2:
            with self._lock:
                self._st["last_error"] = f"exit {rc}, {len(got)} of 3 lines parsed"
                self._st["fallbacks"] += 1
            self._final(rec["id"], "rule", "rule-based", rules, findings, timing, [],
                        "Llama's answer could not be used (wrong format); rule-based advice is shown.",
                        raw=split_answer(out))
            return
        allowed = set(NUM.findall(user))
        lines, guard = {}, []
        for k in KEYS:
            v = clip(got[k]) if k in got else None
            if v is None:
                lines[k] = rules[k]
            elif any(n not in allowed for n in NUM.findall(v)):
                lines[k] = rules[k]
                guard.append(k)
            else:
                lines[k] = v
        missing = [k for k in KEYS if k not in got]
        replaced = sorted(set(guard) | set(missing))
        with self._lock:
            s = self._st
            s["runs"] += 1
            s["guard_hits"] += len(guard)
            if timing:
                for a, b in (("load_s", "load_s"), ("first_token_s", "first_token_s"),
                             ("tokens_per_s", "tokens_per_s"), ("total_s", "total_s")):
                    if timing.get(b) is not None:
                        s[a].append(timing[b])
            rss = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
            s["peak_rss_mb"] = max(rss, s["peak_rss_mb"] or 0)
        all_llama = not replaced
        source = "llama" if all_llama else "llama (partly rule-based)"
        note = ("Written by Llama on this device." if all_llama else
                "Lines replaced by rule-based text: " + ", ".join(replaced) +
                " (missing, or contained a number that was not in the findings).")
        if all_llama:
            with self._lock:
                self._cache[key] = {"id": rec["id"], "lines": lines, "timing": timing, "guard": guard}
        self._final(rec["id"], "ready", source, lines, findings, timing, guard if guard else [], note,
                    raw=None if all_llama else split_answer(out))
