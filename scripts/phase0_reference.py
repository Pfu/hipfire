#!/usr/bin/env python3
"""Phase 0 du plan qwen35 tp2 q8 : scelle les artefacts, mesure le peer access
et les baselines de debit (mq4 tp2 et pp2), puis redige le compte rendu
horodate attendu comme porte de sortie de la phase.

Le script pilote le binaire `hipfire` sous le verrou GPU du plan et n'ouvre
jamais HIP lui-meme. Chaque commande est loguee integralement (stdout, stderr,
code retour) sous `<out>/run-<horodatage>/logs/`, les sorties `bench -j` brutes
sous `benches/`, et le compte rendu synthetise dans `report.md` accompagne de
`summary.json`. Code retour 0 si toutes les etapes passent, 1 sinon, 2 en cas
d'erreur d'usage.

Le script ne telecharge jamais sans `--pull` : si l'artefact mq4 vise est deja present dans
models/ il est scelle tel quel (mq4, mq4-xts, mq4-pro... au choix via --mq4-file), sinon le
script refuse en listant les candidats trouves. LesTelechargements `--pull` passent par
`hipfire pull`, qui lit lui-meme HF_TOKEN ; le script n'affiche jamais la valeur du jeton.

Exemples :
  python3 scripts/phase0_reference.py
  python3 scripts/phase0_reference.py --mq4-file qwen3.8-27b.mq4-xts --processes 3
  python3 scripts/phase0_reference.py --pull --processes 3
  python3 scripts/phase0_reference.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PEER_ACCESS_RE = re.compile(r"dense qwen TP load complete: (\d+) ranks, peer_access=(\S+)")
ANOMALY_RE = re.compile(r"load failed|refused|error: ", re.IGNORECASE)
STAT_KEYS = ("mean", "median", "min", "max")
GIB = 1024**3


@dataclass
class StepResult:
    name: str
    displayed: str
    ok: bool
    exit_code: int | None
    log_path: str | None
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bin", dest="hipfire_bin", type=Path, default=Path("./target/release/hipfire"),
                        help="chemin du binaire hipfire (defaut ./target/release/hipfire)")
    parser.add_argument("--trunk", type=Path, default=Path("~/qwen3.8-27b.q8.hfq"),
                        help="chemin du trunk q8 a sceller")
    parser.add_argument("--mq4-tag", default="qwen3.8:27b",
                        help="tag registre de la reference mq4 (defaut qwen3.8:27b)")
    parser.add_argument("--mq4-file", default=None,
                        help="nom du fichier mq4 dans models/ (sinon lu via `registry show`)")
    parser.add_argument("--mtp-file", default=None,
                        help="nom du fichier de la tete MTP (sinon derive du registre)")
    parser.add_argument("--out", type=Path, default=Path("phase0-out"),
                        help="repertoire de sortie (defaut ./phase0-out)")
    parser.add_argument("--flock", default="/tmp/hipfire-gpu.lock",
                        help="chemin du verrou GPU (defaut /tmp/hipfire-gpu.lock)")
    parser.add_argument("--flock-wait", default="3600",
                        help="attente maximale du verrou en secondes (defaut 3600)")
    parser.add_argument("--processes", type=int, default=3,
                        help="processus frais par baseline (defaut 3)")
    parser.add_argument("--tg", type=int, default=128, help="tokens de decode par run bench (defaut 128)")
    parser.add_argument("--max-seq", type=int, default=4096, help="contexte des runs de peer access")
    parser.add_argument("--prompt", default="Hello", help="prompt des runs de peer access")
    parser.add_argument("--timeout", type=int, default=3600, help="timeout par commande en secondes")
    parser.add_argument("--pull", action="store_true",
                        help="autoriser `hipfire pull` si l'artefact mq4 vise est absent "
                             "(sans ce drapeau, jamais de telechargement)")
    parser.add_argument("--dry-run", action="store_true", help="afficher les commandes sans rien executer")
    return parser.parse_args()


def display_command(cmd: list[str], env_extra: dict[str, str] | None = None) -> str:
    prefix = " ".join(f"{key}={value}" for key, value in (env_extra or {}).items())
    command = " ".join(shlex.quote(part) for part in cmd)
    return f"{prefix} {command}".strip()


def run_step(args: argparse.Namespace, name: str, cmd: list[str], log_path: Path | None,
             env_extra: dict[str, str] | None = None,
             acceptable_exits: tuple[int, ...] = (0,)) -> StepResult:
    full = ["flock", "-w", args.flock_wait, args.flock] + cmd
    displayed = display_command(full, env_extra)
    if args.dry_run:
        print(f"[dry-run] {displayed}")
        return StepResult(name, displayed, True, None, None, "dry-run")
    env = os.environ.copy()
    env.update(env_extra or {})
    try:
        proc = subprocess.run(full, capture_output=True, text=True, env=env, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"$ {displayed}\n--- timeout after {args.timeout}s ---\n")
        return StepResult(name, displayed, False, None, str(log_path), f"timeout after {args.timeout}s")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {displayed}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
            f"--- exit {proc.returncode} ---\n"
        )
    ok = proc.returncode in acceptable_exits
    detail = "" if ok else f"exit {proc.returncode} (acceptes : {', '.join(map(str, acceptable_exits))})"
    print(f"[{'ok' if ok else 'FAIL'}] {name} ({displayed})")
    return StepResult(name, displayed, ok, proc.returncode, str(log_path) if log_path else None, detail)


def log_stdout(log_path: str | Path | None) -> str:
    if log_path is None:
        return ""
    path = Path(log_path)
    if not path.is_file():
        return ""
    text = path.read_text()
    marker = "--- stdout ---\n"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("\n--- stderr ---", 1)[0]


def full_log(log_path: str | Path | None) -> str:
    if log_path is None:
        return ""
    path = Path(log_path)
    return path.read_text() if path.is_file() else ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "inconnu"
    return f"{value / GIB:.1f} GiB"


def models_dir() -> Path:
    home = os.environ.get("HIPFIRE_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".hipfire") / "models"


def find_topology(obj: Any) -> Any:
    if isinstance(obj, dict):
        if {"pp", "tp", "ep", "cards"} <= set(obj):
            return obj
        for value in obj.values():
            found = find_topology(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_topology(value)
            if found is not None:
                return found
    return None


def extract_peer_access(text: str) -> dict[str, Any] | None:
    match = PEER_ACCESS_RE.search(text)
    if match is None:
        return None
    return {"ranks": int(match.group(1)), "peer_access": match.group(2), "line": match.group(0)}


def lines_matching(text: str, needle: str, limit: int = 20) -> list[str]:
    return [line.strip() for line in text.splitlines() if needle in line.lower()][:limit]


def anomaly_lines(text: str, limit: int = 10) -> list[str]:
    return [line.strip() for line in text.splitlines() if ANOMALY_RE.search(line)][:limit]


def parse_bench_report(report: dict[str, Any]) -> dict[str, Any]:
    raw_samples = report.get("samples")
    samples = raw_samples if isinstance(raw_samples, dict) else {}

    def stats(key: str) -> dict[str, Any] | None:
        block = report.get(key)
        if not isinstance(block, dict):
            return None
        return {name: block.get(name) for name in STAT_KEYS if name in block}

    return {
        "decode_tok_s": stats("decode_tok_s"),
        "prefill_tok_s": stats("prefill_tok_s"),
        "decode_samples": samples.get("decode") or [],
        "prefill_samples": samples.get("prefill") or [],
        "prompt_tokens": report.get("prompt_tokens"),
        "max_seq": report.get("max_seq"),
        "topology": find_topology(report),
        "warnings": report.get("warnings") or [],
    }


def pooled(values: list[Any]) -> dict[str, Any] | None:
    flat = sorted(value for value in values if isinstance(value, (int, float)))
    if not flat:
        return None
    count = len(flat)
    middle = count // 2
    median = flat[middle] if count % 2 else (flat[middle - 1] + flat[middle]) / 2
    return {
        "n": count,
        "mean": round(sum(flat) / count, 2),
        "median": round(median, 2),
        "min": round(flat[0], 2),
        "max": round(flat[-1], 2),
    }


def fit_machine(report: dict[str, Any]) -> dict[str, Any]:
    raw_machine = report.get("machine")
    machine = raw_machine if isinstance(raw_machine, dict) else {}
    devices = []
    for device in machine.get("devices") or []:
        devices.append({
            "index": device.get("index"),
            "arch": device.get("arch"),
            "vram_total": gib(device.get("vram_total")),
            "free": gib(device.get("free")),
            "link": device.get("link"),
        })
    raw_plan = report.get("plan")
    plan = raw_plan if isinstance(raw_plan, dict) else {}
    return {
        "devices": devices,
        "ram_total": gib(machine.get("ram_total")),
        "mem_available": gib(machine.get("mem_available")),
        "plan_topology": plan.get("topology"),
    }


def seal_artifact(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path), "present": False}
    return {"path": str(path), "present": True, "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path)}


def build_report(context: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 0 — cadrage et mesures de reference (qwen35 tp2 q8)")
    lines.append("")
    lines.append(f"- Date (UTC) : {context['timestamp']}")
    lines.append(f"- Binaire : {context['binary_version']}")
    lines.append(f"- Prompt : \"{context['prompt']}\" (greedy, `--temp 0`)")
    lines.append(f"- Verrou GPU : {context['flock']}")
    lines.append("")
    lines.append("## Machine (hipfire fit, plan tp2)")
    machine = context["machine"]
    for device in machine["devices"]:
        link = f", lien {device['link']}" if device.get("link") else ""
        lines.append(f"- GPU{device['index']} {device['arch']} : {device['free']} libres / "
                     f"{device['vram_total']}{link}")
    lines.append(f"- RAM {machine['ram_total']}, disponible {machine['mem_available']}")
    lines.append(f"- fit plan topology : {machine['plan_topology']}")
    lines.append("")
    lines.append("## Artefacts scelles")
    for name, artifact in context["artifacts"].items():
        if not artifact.get("present"):
            lines.append(f"- {name} : ABSENT ({artifact['path']})")
            continue
        size = artifact.get("size_bytes")
        size_text = f"{size / GIB:.2f} GiB" if isinstance(size, (int, float)) else "?"
        lines.append(f"- {name} : {artifact['path']} ({size_text}, sha256 {artifact.get('sha256')})")
    pull = context["pull"]
    lines.append(f"- telechargement : {pull['note']}")
    token_text = "presente" if pull["hf_token_present"] else "absente"
    lines.append(f"- HF_TOKEN : {token_text} (lue par hipfire lui-meme pour le depot HF "
                 f"`{pull['repo'] or 'inconnu'}` ; valeur jamais affichee)")
    if context["partials"]:
        lines.append("- fichiers partiels presents dans models/ (pulls interrompus, non geres ici) :")
        for name in context["partials"]:
            lines.append(f"  - {name}")
    lines.append("")
    lines.append("## Peer access (run mq4 tp2)")
    peer = context["peer_access"]
    if peer is None:
        lines.append("- ligne `dense qwen TP load complete` NON TROUVEE — voir le log brut")
    else:
        lines.append(f"- ligne : `{peer['line']}`")
        lines.append(f"- peer_access : {peer['peer_access']} ({peer['ranks']} ranks)")
        lines.append(f"- source de la ligne : {peer.get('source', 'run-mq4-tp2')}")
    for line in context["peer_anomalies"]:
        lines.append(f"- anomalie : {line}")
    lines.append("")
    lines.append("## Attache MTP (run mq4 tp2 --spec mtp)")
    if context["mtp_lines"]:
        for line in context["mtp_lines"]:
            lines.append(f"- {line}")
    else:
        lines.append("- aucune ligne mtp dans la sortie — voir le log brut")
    for line in context["mtp_anomalies"]:
        lines.append(f"- anomalie : {line}")
    lines.append("")
    lines.append(f"## Baselines de debit ({context['processes']} processus frais, "
                 f"tg{context['tg']})")
    for config in ("tp2", "pp2"):
        data = context["baselines"][config]
        lines.append(f"### mq4 {config}")
        for index, run in enumerate(data["runs"], start=1):
            decode = run.get("decode_tok_s")
            prefill = run.get("prefill_tok_s")
            decode_text = f"decode mean {decode['mean']}" if decode else "decode n/a"
            prefill_text = f"prefill mean {prefill['mean']}" if prefill else "prefill n/a"
            lines.append(f"- processus {index} : {decode_text} tok/s ; {prefill_text} tok/s ; "
                         f"topology {run.get('topology')}")
            for warning in run.get("warnings") or []:
                lines.append(f"  - warning : {warning}")
        lines.append(f"- pool decode : {data['pool_decode']}")
        lines.append(f"- pool prefill : {data['pool_prefill']}")
        lines.append("")
    lines.append("## Environnement HIPFIRE_* actif")
    for key, value in context["hipfire_env"].items():
        lines.append(f"- {key}={value}")
    lines.append("")
    lines.append("## Anomalies (toutes etapes)")
    if context["anomalies"]:
        for line in context["anomalies"]:
            lines.append(f"- {line}")
    else:
        lines.append("- aucune")
    lines.append("")
    lines.append("## Etapes")
    for step in context["steps"]:
        status = "ok" if step["ok"] else "FAIL"
        lines.append(f"- {status} {step['name']} (exit {step['exit_code']}) : {step['detail']}")
    lines.append("")
    lines.append("## Logs bruts")
    for log in context["log_files"]:
        lines.append(f"- {log}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    hipfire = args.hipfire_bin.expanduser()
    trunk = args.trunk.expanduser()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / f"run-{stamp}"
    logs_dir = run_dir / "logs"
    benches_dir = run_dir / "benches"
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        benches_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("flock") is None:
            print("flock introuvable (paquet util-linux) ; installez-le avant d'executer la phase 0",
                  file=sys.stderr)
            return 2
        if not hipfire.is_file():
            print(f"binaire hipfire introuvable : {hipfire} (construisez-le ou passez --bin)",
                  file=sys.stderr)
            return 2

    steps: list[StepResult] = []
    version = run_step(args, "version", [str(hipfire), "--version"], logs_dir / "version.log")
    steps.append(version)
    version_lines = [line for line in log_stdout(version.log_path).splitlines() if line.strip()]
    version_text = version_lines[0] if version_lines else ""

    show = run_step(args, "registry-show",
                    [str(hipfire), "registry", "show", args.mq4_tag, "-j"],
                    logs_dir / "registry-show.log")
    steps.append(show)
    mq4_name, mtp_name = args.mq4_file, args.mtp_file
    repo = ""
    if not args.dry_run and show.ok:
        try:
            entry = json.loads(log_stdout(show.log_path) or "{}")
        except json.JSONDecodeError as error:
            print(f"registry show illisible ({error}) ; passez --mq4-file/--mtp-file", file=sys.stderr)
            entry = {}
        if isinstance(entry, dict):
            mq4_name = mq4_name or entry.get("file")
            mtp_block = entry.get("mtp")
            mtp_name = mtp_name or (mtp_block.get("file") if isinstance(mtp_block, dict) else None)
            repo = entry.get("repo") or ""
    mq4_path = (models_dir() / mq4_name) if mq4_name else None
    if mtp_name is None and mq4_name:
        mtp_name = str(Path(mq4_name).with_suffix(".mtp"))
    mtp_path = (models_dir() / mtp_name) if mtp_name else None

    pull_note = "non execute (--pull absent) ; aucun telechargement"
    pull_executed = False
    hf_token_present = bool(os.environ.get("HF_TOKEN"))
    if args.dry_run:
        print("[dry-run] presence des artefacts verifiee seulement en execution reelle ; "
              f"pull : {'serait execute (--pull)' if args.pull else 'jamais (--pull absent)'}")
    elif mq4_path and mq4_path.is_file() and mtp_path and mtp_path.is_file():
        pull_note = "artefacts deja presents ; aucun telechargement"
        print(f"[skip] pull ({pull_note} : {mq4_path.name}, {mtp_path.name})")
    elif args.pull:
        steps.append(run_step(args, "pull", [str(hipfire), "pull", args.mq4_tag, "-y"],
                              logs_dir / "pull.log"))
        pull_executed = True
        pull_note = f"execute via `hipfire pull {args.mq4_tag}` (depot HF : {repo or 'inconnu'})"
        if not (mq4_path and mq4_path.is_file()):
            print(f"fichier mq4 toujours absent apres pull : {mq4_path}", file=sys.stderr)
            return 2
    else:
        print(f"artefact mq4 absent : {mq4_path} ; le script ne telecharge pas sans --pull",
              file=sys.stderr)
        for candidate in sorted(models_dir().glob("*.mq4*")):
            print(f"  candidat : {candidate.name} ({candidate.stat().st_size / GIB:.2f} GiB) -> "
                  f"relancez avec --mq4-file {candidate.name}", file=sys.stderr)
        return 2

    artifacts: dict[str, dict[str, Any]] = {}
    if args.dry_run:
        artifacts["trunk_q8"] = {"path": str(trunk), "present": None}
        artifacts["mq4"] = {"path": str(mq4_path or models_dir() / f"<fichier de {args.mq4_tag}>"),
                            "present": None}
        artifacts["mtp"] = {"path": str(mtp_path or models_dir() / "<derive du registre>"),
                            "present": None}
    else:
        artifacts["trunk_q8"] = seal_artifact(trunk)
        artifacts["mq4"] = seal_artifact(mq4_path)
        artifacts["mtp"] = seal_artifact(mtp_path)
        for name, artifact in artifacts.items():
            status = "ok" if artifact.get("present") else "ABSENT"
            print(f"[{status}] {name}: {artifact['path']}")

    model_arg = str(mq4_path) if mq4_path else str(models_dir() / f"<fichier de {args.mq4_tag}>")
    run_cmd = [str(hipfire), "run", model_arg, "--tp", "2", "--max-seq", str(args.max_seq),
               "--temp", "0", args.prompt]
    peer_run = run_step(args, "run-mq4-tp2", run_cmd, logs_dir / "run-mq4-tp2.log")
    steps.append(peer_run)
    peer_text = full_log(peer_run.log_path)
    peer_access = extract_peer_access(peer_text)
    peer_source = "run-mq4-tp2"
    if peer_access is not None:
        peer_access["source"] = peer_source

    mtp_cmd = run_cmd.copy()
    mtp_index = mtp_cmd.index("--tp")
    mtp_cmd[mtp_index:mtp_index] = ["--spec", "mtp"]
    mtp_run = run_step(args, "run-mq4-tp2-mtp", mtp_cmd, logs_dir / "run-mq4-tp2-mtp.log")
    steps.append(mtp_run)
    mtp_text = full_log(mtp_run.log_path)
    if peer_access is None:
        fallback = extract_peer_access(mtp_text)
        if fallback is not None:
            peer_access = fallback
            fallback["source"] = "run-mq4-tp2-mtp (run sans --spec echoue avant le chargement complet)"

    baselines: dict[str, dict[str, Any]] = {}
    for config, env_extra in (("tp2", None), ("pp2", {"HIPFIRE_TOPOLOGY_PP": "2"})):
        runs: list[dict[str, Any]] = []
        for index in range(1, args.processes + 1):
            bench_path = benches_dir / f"p0-mq4-{config}-{index}.json"
            bench_log = logs_dir / f"bench-mq4-{config}-{index}.log"
            bench_cmd = [str(hipfire), "bench", model_arg, "--tg", str(args.tg), "-j"]
            step = run_step(args, f"bench-mq4-{config}-{index}", bench_cmd, bench_log,
                            env_extra=env_extra)
            steps.append(step)
            parsed: dict[str, Any] = {"ok": step.ok}
            stdout = log_stdout(bench_log).strip()
            if step.ok and stdout:
                bench_path.write_text(stdout + "\n")
                parsed.update(parse_bench_report(json.loads(stdout)))
            runs.append(parsed)
        pool_decode = pooled([value for run in runs for value in (run.get("decode_samples") or [])])
        pool_prefill = pooled([value for run in runs for value in (run.get("prefill_samples") or [])])
        baselines[config] = {"runs": runs, "pool_decode": pool_decode, "pool_prefill": pool_prefill}

    fit_report: dict[str, Any] = {}
    if not args.dry_run:
        fit_step = run_step(args, "fit-mq4-tp2",
                            [str(hipfire), "fit", model_arg, "--tp", "2", "-j"],
                            logs_dir / "fit-mq4-tp2.json.log", acceptable_exits=(0, 1, 2))
        steps.append(fit_step)
        if fit_step.ok:
            stdout = log_stdout(fit_step.log_path).strip()
            if stdout:
                parsed_fit = json.loads(stdout)
                fit_report = fit_machine(parsed_fit)
                verdict = parsed_fit.get("verdict")
                if verdict is not None:
                    fit_step.detail = f"verdict={verdict} (exit fit {fit_step.exit_code})"
    if not fit_report:
        fit_report = {"devices": [], "ram_total": "n/a", "mem_available": "n/a",
                      "plan_topology": "n/a"}

    partials = sorted(models_dir().glob("*.part.*")) if not args.dry_run else []
    log_files = sorted(logs_dir.glob("*.log")) if logs_dir.is_dir() else []
    all_logs = "\n".join(path.read_text() for path in log_files)
    anomalies = sorted(set(anomaly_lines(all_logs)))
    steps_json = [{"name": step.name, "ok": step.ok, "exit_code": step.exit_code,
                   "detail": step.detail,
                   "anomalies": anomaly_lines(full_log(step.log_path))}
                  for step in steps]
    summary = {
        "timestamp": stamp,
        "binary_version": version_text,
        "prompt": args.prompt,
        "artifacts": artifacts,
        "peer_access": peer_access,
        "mtp_lines": lines_matching(mtp_text, "mtp"),
        "baselines": baselines,
        "fit_machine": fit_report,
        "pull": {"executed": pull_executed, "note": pull_note,
                 "hf_token_present": hf_token_present, "repo": repo},
        "partials": [path.name for path in partials],
        "steps": steps_json,
    }
    report_text = build_report({
        "timestamp": stamp,
        "binary_version": version_text or "n/a",
        "prompt": args.prompt,
        "flock": args.flock,
        "processes": args.processes,
        "tg": args.tg,
        "machine": fit_report,
        "artifacts": artifacts,
        "peer_access": peer_access,
        "peer_anomalies": anomaly_lines(peer_text),
        "mtp_lines": summary["mtp_lines"],
        "mtp_anomalies": anomaly_lines(mtp_text),
        "pull": summary["pull"],
        "partials": summary["partials"],
        "baselines": baselines,
        "hipfire_env": {key: value for key, value in os.environ.items()
                        if key.startswith("HIPFIRE_")},
        "anomalies": anomalies,
        "steps": steps_json,
        "log_files": [str(path) for path in log_files],
    })
    if not args.dry_run:
        (run_dir / "report.md").write_text(report_text)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\ncompte rendu : {run_dir / 'report.md'}")
        print(f"resume machine : {run_dir / 'summary.json'}")
        failed = [step for step in steps if not step.ok]
        if failed:
            print(f"etapes en echec : {', '.join(step.name for step in failed)}", file=sys.stderr)
            for step in failed:
                for line in anomaly_lines(full_log(step.log_path), limit=5):
                    print(f"  [{step.name}] {line}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
