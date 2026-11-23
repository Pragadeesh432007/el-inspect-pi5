"""Quick test of the Llama advisory without the dashboard.

Sends a few made-up findings to Llama (ExecuTorch llama_main) exactly as the app does and prints what Llama
really said, whether the three lines were parsed, the number guard result, and the timing. Use it to tune the
prompt and to collect the Llama numbers for the report.

    python llama_selftest.py                       # all cases, default folder ~/executorch-deployment
    python llama_selftest.py --case 2 --repeat 3   # one case, three times
    python llama_selftest.py --temperature 0.8     # sampling like llama_main's default
"""
import argparse
import re
from pathlib import Path

from advisor import (Advisor, NUM, build_findings, build_prompt, extract_lines, parse_observer, rule_lines,
                     split_answer)


def rec(i, sev, p, faults, action, status="DEFECTIVE"):
    return {"id": f"TEST-{i}", "status": status, "severity": sev, "p_def": p, "action": action,
            "faults": [{"key": k, "name": n, "desc": d, "p": q} for k, n, d, q in faults], "quality": {"issues": []}}


CASES = [
    rec(1, "moderate", 0.91, [("crack", "Crack", "micro-crack in the silicon", 0.91)],
        "Schedule a detailed inspection or repair."),
    rec(2, "severe", 0.99, [("black_core", "Black core", "dark inactive region (power loss)", 0.95)],
        "Remove from service. Repair or replace soon."),
    rec(3, "severe", 0.99, [("short_circuit", "Short circuit", "short-circuited / dead cell area", 0.93),
                            ("other", "Other", "other defect", 0.60)],
        "Remove from service. Repair or replace soon."),
    rec(4, "low", 0.97, [("finger", "Finger defect", "broken or missing finger electrode", 0.96)],
        "Monitor. Re-inspect at the next scheduled check."),
    rec(5, "moderate", 0.88, [], "Schedule a detailed inspection or repair. (defect type unclear: inspect manually)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dir", default="~/executorch-deployment")
    ap.add_argument("--llama-model", default="Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte")
    ap.add_argument("--llama-tokenizer", default="tokenizer.model")
    ap.add_argument("--case", type=int, help="run only this case number (1-5)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic (default)")
    a = ap.parse_args()
    adv = Advisor(a.llama_dir, a.llama_model, a.llama_tokenizer, lambda x: None,
                  log_path="/dev/null" if Path("/dev/null").exists() else "selftest_log.jsonl",
                  temperature=a.temperature)
    if not adv.available:
        raise SystemExit(adv.reason)
    ok = total = 0
    for n, r in enumerate(CASES, 1):
        if a.case and n != a.case:
            continue
        findings = build_findings(r)
        prompt, user = build_prompt(findings)
        for k in range(a.repeat):
            out, err, rc = adv._run_llama(prompt)
            ans, got, t = split_answer(out), extract_lines(split_answer(out)), parse_observer(out, err)
            bad = [x for x, v in got.items() if any(m not in set(NUM.findall(user)) for m in NUM.findall(v))]
            good = len(got) == 3 and not bad
            ok, total = ok + good, total + 1
            print(f"\n=== case {n} run {k + 1}: {[d['type'] for d in findings['defects']] or 'type unclear'} "
                  f"| severity {findings['severity']} ===")
            print(ans if ans else "(empty answer)")
            print(f"-> parsed {len(got)}/3 lines, numbers not in findings: {bad or 'none'}, usable: {'YES' if good else 'NO'}")
            if t:
                print(f"-> prompt {t['prompt_tokens']} tok | load {t['load_s']} s | first token {t['first_token_s']} s | "
                      f"{t['generated_tokens']} tok at {t['tokens_per_s']} tok/s | total {t['total_s']} s")
    print(f"\nUsable answers: {ok}/{total}")


if __name__ == "__main__":
    main()
