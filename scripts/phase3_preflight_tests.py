#!/usr/bin/env python3
"""Compil et tests du crate hipfire-arch-qwen35 (porte de format qt3 du TP
dense) avec resume d'execution pour analyse.

Le script enchaîne, depuis la racine du depot :
  1. contexte : git rev/branche/etat, rustc --version, cargo --version ;
  2. compilation des tests : cargo test -p hipfire-arch-qwen35 --no-run
     (warnings et erreurs captures) ;
  3. execution : cargo test -p hipfire-arch-qwen35 -- --show-output
     (l'option rend visibles les eprintln "skip: ..." des tests d'integration
     qui retournent tot sans GPU ou sans fixture, et les compte) ;
  4. redaction sous <out>/run-<horodatage>/ : logs bruts par commande (logs/),
     report.md (resume lisible) et summary.json (resume machine). La console
     affiche un bloc final compact, pret a coller dans la session d'analyse.

Par defaut les commandes cargo passent sous le verrou GPU du plan (les tests
d'integration peuvent toucher le GPU ; les suites *_hw sont #[ignore] et ne
s'executent pas). --lib-only limite l'execution aux tests unitaires de la
lib (aucun GPU), --filter passe un filtre de nom de test apres `--`.

Code retour 0 si compilation et tests passent, 1 en cas d'echec, 2 en erreur
d'usage.

Exemples :
  python3 scripts/phase3_preflight_tests.py
  python3 scripts/phase3_preflight_tests.py --lib-only
  python3 scripts/phase3_preflight_tests.py --filter dense_tp_preflight_format_tests
  python3 scripts/phase3_preflight_tests.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CRATE = "hipfire-arch-qwen35"
PREFLIGHT_MOD = "dense_tp_preflight_format_tests"

SUITE_HEADER_RE = re.compile(r"^\s*(Running\s+\S+.*|Doc-tests\s+\S+.*)$", re.MULTILINE)
TEST_LINE_RE = re.compile(
    r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)(?:\s+\(([\d.]+)s\))?",
    re.MULTILINE,
)
RESULT_LINE_RE = re.compile(
    r"^test result:\s+(ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored;"
    r"\s+(\d+) measured;\s+(\d+) filtered out;\s+finished in\s+([\d.]+)s",
    re.MULTILINE,
)
STDOUT_BLOCK_RE = re.compile(r"^----\s+(.+?)\s+stdout\s+----\s*$", re.MULTILINE)
SKIP_RE = re.compile(r"^skip:\s*(.*)$", re.MULTILINE)
GPU_SKIP_RE = re.compile(r"no gpu|gpu absent|init", re.IGNORECASE)


@dataclass
class StepResult:
    name: str
    displayed: str
    ok: bool
    exit_code: int | None
    log_path: str | None
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class TestSuite:
    header: str
    result: str = "n/a"
    passed: int = 0
    failed: int = 0
    ignored: int = 0
    measured: int = 0
    filtered_out: int = 0
    finished_s: float = 0.0
    tests: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("phase3-out"),
                        help="repertoire de sortie (defaut ./phase3-out)")
    parser.add_argument("--lib-only", action="store_true",
                        help="executer seulement les tests unitaires de la lib "
                             "(cargo test --lib ; aucun GPU requis)")
    parser.add_argument("--filter", default=None,
                        help="filtre de nom de test passe au binaire (apres --)")
    parser.add_argument("--features", default=None,
                        help="features cargo supplementaires (liste separee par des virgules)")
    parser.add_argument("--release", action="store_true",
                        help="compiler et executer en release (defaut : debug)")
    parser.add_argument("--no-show-output", action="store_true",
                        help="ne pas passer --show-output (les skips runtime ne seront "
                             "plus comptes)")
    parser.add_argument("--flock", default="/tmp/hipfire-gpu.lock",
                        help="chemin du verrou GPU (defaut /tmp/hipfire-gpu.lock)")
    parser.add_argument("--flock-wait", default="3600",
                        help="attente maximale du verrou en secondes (defaut 3600)")
    parser.add_argument("--no-flock", action="store_true",
                        help="ne pas prendre le verrou GPU (machine sans GPU)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="timeout par commande en secondes (defaut 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="afficher les commandes sans rien executer")
    return parser.parse_args()


def display_command(cmd: list[str], env_extra: dict[str, str] | None = None) -> str:
    prefix = " ".join(f"{key}={value}" for key, value in (env_extra or {}).items())
    command = " ".join(shlex.quote(part) for part in cmd)
    return f"{prefix} {command}".strip()


def run_step(args: argparse.Namespace, name: str, cmd: list[str], log_path: Path | None,
             env_extra: dict[str, str] | None = None,
             use_flock: bool = False) -> StepResult:
    full = cmd
    if use_flock and not args.no_flock:
        full = ["flock", "-w", args.flock_wait, args.flock] + full
    displayed = display_command(full, env_extra)
    if args.dry_run:
        print(f"[dry-run] {displayed}")
        return StepResult(name, displayed, True, None, None, "dry-run")
    env = os.environ.copy()
    env.update(env_extra or {})
    started = time.monotonic()
    try:
        proc = subprocess.run(full, capture_output=True, text=True, env=env,
                              timeout=args.timeout, cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - started
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"$ {displayed}\n--- timeout after {args.timeout}s ---\n")
        print(f"[FAIL] {name} (timeout {args.timeout}s)")
        return StepResult(name, displayed, False, None, str(log_path),
                          f"timeout after {args.timeout}s", duration)
    duration = time.monotonic() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {displayed}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
            f"--- exit {proc.returncode} ---\n"
        )
    ok = proc.returncode == 0
    print(f"[{'ok' if ok else 'FAIL'}] {name} ({duration:.1f}s)")
    return StepResult(name, displayed, ok, proc.returncode,
                      str(log_path) if log_path else None, "", duration)


def full_log(log_path: str | Path | None) -> str:
    if log_path is None:
        return ""
    path = Path(log_path)
    return path.read_text() if path.is_file() else ""


def split_log_sections(log_path: str | None) -> tuple[str, str]:
    text = full_log(log_path)
    if not text:
        return "", ""
    stdout_part = text.split("--- stdout ---\n", 1)[1].split("\n--- stderr ---", 1)[0] \
        if "--- stdout ---\n" in text else ""
    stderr_part = text.split("\n--- stderr ---\n", 1)[1].split("\n--- exit ", 1)[0] \
        if "\n--- stderr ---\n" in text else ""
    return stdout_part, stderr_part


def parse_cargo_diagnostics(stderr: str) -> dict[str, Any]:
    raw_warnings = [line.strip() for line in stderr.splitlines()
                    if line.startswith("warning: ")]
    warning_lines = [line for line in raw_warnings
                     if "generated" not in line or " warnings" not in line and " warning" not in line
                     or not re.match(r"warning:\s+`[\w-]+`\s+\([^)]*\)\s+generated", line)]
    error_lines = [line.strip() for line in stderr.splitlines()
                   if line.startswith("error")]
    generated = re.findall(
        r"warning:\s+`([\w-]+)`\s+\(([^)]*)\)\s+generated\s+(\d+)\s+warnings?",
        stderr)
    finished = next((line.strip() for line in stderr.splitlines()
                     if line.strip().startswith("Finished")), "")
    return {
        "warning_count": len(warning_lines),
        "warning_lines": warning_lines[:20],
        "error_lines": error_lines[:20],
        "generated_by_crate": [
            {"crate": crate, "unit": unit, "warnings": int(count)}
            for crate, unit, count in generated
        ],
        "finished_line": finished,
    }


def parse_libtest(stdout: str) -> dict[str, Any]:
    headers = list(SUITE_HEADER_RE.finditer(stdout))
    suites: list[TestSuite] = []
    for index, match in enumerate(headers):
        start = match.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(stdout)
        body = stdout[start:end]
        suite = TestSuite(header=match.group(0).strip())
        result = RESULT_LINE_RE.search(body)
        if result is not None:
            suite.result = result.group(1)
            suite.passed = int(result.group(2))
            suite.failed = int(result.group(3))
            suite.ignored = int(result.group(4))
            suite.measured = int(result.group(5))
            suite.filtered_out = int(result.group(6))
            suite.finished_s = float(result.group(7))
        suite.tests = [
            {"name": name, "status": status, "duration_s": float(duration) if duration else None}
            for name, status, duration in TEST_LINE_RE.findall(body)
        ]
        suites.append(suite)
    failure_blocks: dict[str, str] = {}
    blocks = list(STDOUT_BLOCK_RE.finditer(stdout))
    for index, match in enumerate(blocks):
        start = match.end()
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(stdout)
        body = stdout[start:end]
        cut = body.find("\nfailures:")
        if cut != -1:
            body = body[:cut]
        failure_blocks[match.group(1)] = body.strip("\n")
    failed_names = [test["name"] for suite in suites for test in suite.tests
                    if test["status"] == "FAILED"]
    failures = [{"name": name, "output": failure_blocks.get(name, "(sortie absente)")}
                for name in failed_names]
    skip_counts: dict[str, int] = {}
    skip_lines: list[str] = []
    for match in SKIP_RE.finditer(stdout):
        line = match.group(0).strip()
        skip_lines.append(line)
        reason = "gpu" if GPU_SKIP_RE.search(match.group(1)) else "fixture/autre"
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
    preflight = [test["name"] for suite in suites for test in suite.tests
                 if PREFLIGHT_MOD in test["name"]]
    preflight_ok = [test["name"] for suite in suites for test in suite.tests
                    if PREFLIGHT_MOD in test["name"] and test["status"] == "ok"]
    return {
        "suites": [vars(suite) for suite in suites],
        "failures": failures,
        "skip_total": len(skip_lines),
        "skip_counts": skip_counts,
        "skip_lines": skip_lines[:40],
        "preflight_mod": {"total": len(preflight), "ok": len(preflight_ok)},
        "totals": {
            "passed": sum(suite.passed for suite in suites),
            "failed": sum(suite.failed for suite in suites),
            "ignored": sum(suite.ignored for suite in suites),
            "suites": len(suites),
        },
    }


def build_report(context: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Resume compilation + tests — {CRATE}")
    lines.append("")
    lines.append(f"- run : `{context['run_dir']}` ({context['timestamp']})")
    lines.append(f"- git : `{context['git_rev']}` ({context['git_branch']}, {context['git_state']}) "
             f"— {context['git_subject']}")
    lines.append(f"- rustc : {context['rustc_version']} ; cargo : {context['cargo_version']}")
    lines.append(f"- verrou GPU : {context['flock_note']}")
    compile_step = context["compile"]
    lines.append(f"- compilation des tests : {'OK' if compile_step['ok'] else 'ECHEC'} "
                 f"({compile_step['duration_s']:.1f}s)")
    diag = compile_step["diagnostics"]
    if diag["warning_count"]:
        lines.append(f"  - warnings cargo : {diag['warning_count']}")
        for line in diag["warning_lines"][:10]:
            lines.append(f"    - {line}")
    if diag["error_lines"]:
        lines.append("  - erreurs :")
        for line in diag["error_lines"][:10]:
            lines.append(f"    - {line}")
    lines.append("")
    parsed = context["tests"]
    lines.append("## Suites de tests")
    lines.append("")
    for suite in parsed["suites"]:
        status = suite["result"]
        lines.append(
            f"- `{suite['header']}` : {status} — {suite['passed']} passed, "
            f"{suite['failed']} failed, {suite['ignored']} ignored "
            f"({suite['finished_s']:.1f}s)"
        )
        for test in suite["tests"]:
            if test["status"] != "ok":
                lines.append(f"  - {test['status']}: `{test['name']}`")
    lines.append("")
    preflight = parsed["preflight_mod"]
    lines.append(f"## Mod {PREFLIGHT_MOD}")
    lines.append("")
    lines.append(f"- {preflight['ok']}/{preflight['total']} tests ok")
    lines.append("")
    lines.append("## Echecs")
    lines.append("")
    if parsed["failures"]:
        for failure in parsed["failures"]:
            lines.append(f"### `{failure['name']}`")
            lines.append("")
            lines.append("```")
            for out_line in failure["output"].splitlines()[:20]:
                lines.append(out_line)
            lines.append("```")
            lines.append("")
    else:
        lines.append("aucun.")
    lines.append("")
    lines.append("## Skips runtime (sortie des tests reussis, via --show-output)")
    lines.append("")
    lines.append(f"- total : {parsed['skip_total']} "
                 f"({json.dumps(parsed['skip_counts'], ensure_ascii=True)})")
    for skip_line in parsed["skip_lines"][:20]:
        lines.append(f"- {skip_line}")
    lines.append("")
    lines.append("## Journaux")
    lines.append("")
    for log in context["log_files"]:
        lines.append(f"- {log}")
    lines.append("")
    return "\n".join(lines)


def print_paste_block(context: dict[str, Any]) -> None:
    parsed = context["tests"]
    compile_step = context["compile"]
    totals = parsed["totals"]
    print()
    print("=" * 24, "RESUME POUR ANALYSE", "=" * 24)
    print(f"git     : {context['git_rev']} ({context['git_branch']}, {context['git_state']})")
    print(f"rustc   : {context['rustc_version']}")
    print(f"compile : {'OK' if compile_step['ok'] else 'ECHEC'} "
          f"({compile_step['duration_s']:.1f}s, {compile_step['diagnostics']['warning_count']} warnings)")
    for line in compile_step["diagnostics"]["error_lines"][:5]:
        print(f"  {line}")
    if context["tests_step"] is not None:
        print(f"tests   : {totals['passed']} passed, {totals['failed']} failed, "
              f"{totals['ignored']} ignored ({totals['suites']} suites)")
        for suite in parsed["suites"]:
            print(f"  - {suite['header']}: {suite['result']} "
                  f"({suite['passed']}p/{suite['failed']}f/{suite['ignored']}i)")
        preflight = parsed["preflight_mod"]
        print(f"mod {PREFLIGHT_MOD}: {preflight['ok']}/{preflight['total']} ok")
        if parsed["failures"]:
            print("echecs :")
            for failure in parsed["failures"]:
                first = failure["output"].splitlines()[0] if failure["output"] else ""
                print(f"  - {failure['name']}: {first}")
        print(f"skips   : {parsed['skip_total']} "
              f"({json.dumps(parsed['skip_counts'], ensure_ascii=True)})")
    global_status = "OK" if context["all_ok"] else "ECHEC"
    print(f"statut  : {global_status}")
    print(f"detail  : {context['run_dir'] / 'report.md'} ; "
          f"{context['run_dir'] / 'summary.json'}")
    print("=" * 68)


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / f"run-{stamp}"
    logs_dir = run_dir / "logs"
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("cargo") is None:
            print("cargo introuvable (toolchain Rust) ; installez rustup avant d'executer",
                  file=sys.stderr)
            return 2
        if not (REPO_ROOT / "Cargo.toml").is_file():
            print(f"racine du depot introuvable depuis {REPO_ROOT}", file=sys.stderr)
            return 2
        if not args.no_flock and shutil.which("flock") is None:
            print("flock introuvable (paquet util-linux) ; utilisez --no-flock sur une "
                  "machine sans GPU", file=sys.stderr)
            return 2

    context: dict[str, Any] = {
        "timestamp": stamp,
        "run_dir": str(run_dir),
        "flock_note": ("desactive (--no-flock)" if args.no_flock
                       else f"{args.flock} (attente max {args.flock_wait}s)"),
    }

    def quick(name: str, cmd: list[str]) -> tuple[StepResult, str]:
        log = logs_dir / f"{name}.log"
        step = run_step(args, name, cmd, log, use_flock=False)
        return step, full_log(log)

    git_step, git_text = quick("git-rev",
                               ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    subject_step, subject_text = quick("git-subject",
                                       ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%s"])
    branch_step, branch_text = quick("git-branch",
                                     ["git", "-C", str(REPO_ROOT), "branch", "--show-current"])
    status_step, status_text = quick("git-status",
                                     ["git", "-C", str(REPO_ROOT), "status", "--short"])
    rustc_step, rustc_text = quick("rustc-version", ["rustc", "--version"])
    cargo_step, cargo_text = quick("cargo-version", ["cargo", "--version"])

    def first_line(text: str) -> str:
        return next((line for line in text.splitlines() if line.strip()), "")

    context["git_rev"] = first_line(git_text) or "n/a"
    context["git_subject"] = first_line(subject_text) or "n/a"
    context["git_branch"] = first_line(branch_text) or "n/a"
    context["git_state"] = (f"{len(status_text.splitlines())} fichier(s) modifie(s)"
                            if status_text.strip() else "clean")
    context["rustc_version"] = first_line(rustc_text) or "n/a"
    context["cargo_version"] = first_line(cargo_text) or "n/a"

    cargo_common = ["cargo", "test", "-p", CRATE]
    if args.release:
        cargo_common.append("--release")
    if args.features:
        cargo_common += ["--features", args.features]
    if args.lib_only:
        cargo_common.append("--lib")

    compile_cmd = cargo_common + ["--no-run"]
    compile_step = run_step(args, "compile-tests", compile_cmd, logs_dir / "compile.log",
                            use_flock=True)
    _, compile_stderr = split_log_sections(compile_step.log_path)
    context["compile"] = {
        "ok": compile_step.ok,
        "exit_code": compile_step.exit_code,
        "duration_s": compile_step.duration_s,
        "diagnostics": parse_cargo_diagnostics(compile_stderr),
        "command": compile_step.displayed,
    }
    if not compile_step.ok and not args.dry_run:
        context["tests"] = {"suites": [], "failures": [], "skip_total": 0,
                            "skip_counts": {}, "skip_lines": [],
                            "preflight_mod": {"total": 0, "ok": 0},
                            "totals": {"passed": 0, "failed": 0, "ignored": 0,
                                       "suites": 0}}
        context["tests_step"] = None
        context["all_ok"] = False
        log_files = sorted(str(path) for path in logs_dir.glob("*.log"))
        context["log_files"] = log_files
        report = build_report(context)
        (run_dir / "report.md").write_text(report)
        (run_dir / "summary.json").write_text(json.dumps(context, indent=2) + "\n")
        print_paste_block(context)
        return 1

    test_cmd = cargo_common + ["--", "--show-output"]
    if args.no_show_output:
        test_cmd = cargo_common
    if args.filter:
        test_cmd.append(args.filter)
    tests_step = run_step(args, "run-tests", test_cmd, logs_dir / "tests.log",
                          use_flock=True)
    tests_stdout, tests_stderr = split_log_sections(tests_step.log_path)
    parsed = parse_libtest(tests_stdout) if not args.dry_run else {
        "suites": [], "failures": [], "skip_total": 0, "skip_counts": {},
        "skip_lines": [], "preflight_mod": {"total": 0, "ok": 0},
        "totals": {"passed": 0, "failed": 0, "ignored": 0, "suites": 0}}
    if tests_step.ok and not parsed["suites"] and not args.dry_run:
        tests_step.detail = "aucune suite parsee (sortie libtest inattendue)"
    context["tests"] = parsed
    context["tests_step"] = {
        "ok": tests_step.ok,
        "exit_code": tests_step.exit_code,
        "duration_s": tests_step.duration_s,
        "command": tests_step.displayed,
        "detail": tests_step.detail,
        "stderr_tail": tests_stderr.splitlines()[-10:],
    }
    log_files = sorted(str(path) for path in logs_dir.glob("*.log"))
    context["log_files"] = log_files
    context["all_ok"] = bool(compile_step.ok and tests_step.ok
                             and parsed["totals"]["failed"] == 0)
    if not args.dry_run:
        (run_dir / "report.md").write_text(build_report(context))
        (run_dir / "summary.json").write_text(json.dumps(context, indent=2) + "\n")
        print(f"\nrapport : {run_dir / 'report.md'}")
        print_paste_block(context)
    return 0 if context["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
