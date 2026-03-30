#!/usr/bin/env python3
"""
VQE ML batch generation script (scripts/generate_ml_batch.py).

Generates the Tier 1 (or other tier) VQE dataset by running
molecule x optimizer x init_strategy x seed combinations via
Docker Compose containers.

Architecture:
  - 3 parallel hardware lanes: GPU1, GPU2, CPU
  - GPU1: larger molecules (NH3, CO2, H2O, BeH2)
  - GPU2: medium molecules (N2, LiH)
  - CPU:  small molecules (H2, HeH+)
  - Each lane runs one container at a time (sequential within lane)
  - All three lanes run concurrently via ThreadPoolExecutor
  - State tracked in gen/ml_batch_state.json for idempotent resume

Usage:
  python scripts/generate_ml_batch.py --tier 1            # Generate Tier 1
  python scripts/generate_ml_batch.py --status            # Show progress
  python scripts/generate_ml_batch.py --tier 1 --dry-run  # Preview commands

Tier 1: sto-3g, 8 molecules, 8 optimizers, 2 init strategies, 25 seeds = 3,200 runs
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
STATE_FILE = REPO_ROOT / "gen" / "ml_batch_state.json"
COMPOSE_FILE = REPO_ROOT / "compose" / "docker-compose.ml.yaml"

# Per-lane molecule files (pre-created, mounted into containers at ./data/)
LANE_MOLECULE_FILES = {
    "gpu1": "data/molecules.gpu1.json",   # NH3, CO2, H2O, BeH2
    "gpu2": "data/molecules.gpu2.json",   # N2, LiH
    "cpu":  "data/molecules.cpu.json",    # H2, HeH+
}

# Docker Compose service names matching docker-compose.ml.yaml
LANE_SERVICES = {
    "gpu1": "quantum-pipeline-gpu1",
    "gpu2": "quantum-pipeline-gpu2",
    "cpu":  "quantum-pipeline-cpu",
}

# Whether each lane uses GPU acceleration
LANE_GPU = {
    "gpu1": True,
    "gpu2": True,
    "cpu":  False,
}

# Optimizer phase ordering: run validated optimizers first
OPTIMIZER_PHASES: dict[str, list[str]] = {
    "A": ["L-BFGS-B", "COBYLA", "SLSQP"],
    "B": ["Nelder-Mead", "Powell"],
    "C": ["BFGS", "CG", "TNC"],
}

# ---------------------------------------------------------------------------
# Tier configurations
# ---------------------------------------------------------------------------

TIER_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "basis": "sto3g",
        "optimizers": ["L-BFGS-B", "COBYLA", "SLSQP", "Nelder-Mead", "Powell", "BFGS", "CG", "TNC"],
        "init_strategies": ["random", "hf"],
        "seeds": list(range(1, 26)),      # seeds 1..25
        "ansatz": "EfficientSU2",
        "simulation_method": "statevector",
        "threshold": 1e-6,
        "use_convergence": True,
        "molecule_filter": None,          # all molecules
    },
    2: {
        "basis": "6-31g",
        "optimizers": ["L-BFGS-B", "COBYLA", "SLSQP", "Nelder-Mead", "Powell", "BFGS", "CG", "TNC"],
        "init_strategies": ["random", "hf"],
        "seeds": list(range(1, 16)),      # seeds 1..15
        "ansatz": "EfficientSU2",
        "simulation_method": "statevector",
        "threshold": 1e-6,
        "use_convergence": True,
        "molecule_filter": ["H2", "HeH+", "LiH", "BeH2", "H2O"],
    },
    3: {
        "basis": "sto3g",
        "optimizers": ["L-BFGS-B", "COBYLA", "Nelder-Mead"],
        "init_strategies": ["random", "hf"],
        "seeds": list(range(1, 16)),      # seeds 1..15
        "ansatz_types": ["RealAmplitudes", "ExcitationPreserving"],
        "ansatz": None,                   # iterated over ansatz_types
        "simulation_method": "statevector",
        "threshold": 1e-6,
        "use_convergence": True,
        "molecule_filter": None,
    },
    4: {
        "basis": "cc-pvdz",
        "optimizers": ["L-BFGS-B", "COBYLA", "Nelder-Mead"],
        "init_strategies": ["random", "hf"],
        "seeds": list(range(1, 11)),      # seeds 1..10
        "ansatz": "EfficientSU2",
        "simulation_method": "statevector",
        "threshold": 1e-6,
        "use_convergence": True,
        "molecule_filter": ["H2", "HeH+", "LiH"],
    },
}

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
logger = logging.getLogger(__name__)


def load_state(state_file: Path) -> dict[str, dict]:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(state_file)


def update_run_state(state_file: Path, key: str, updates: dict) -> None:
    with _state_lock:
        state = load_state(state_file)
        if key not in state:
            state[key] = {}
        state[key].update(updates)
        save_state(state_file, state)


# ---------------------------------------------------------------------------
# Run key
# ---------------------------------------------------------------------------

def make_run_key(
    lane: str,
    optimizer: str,
    init_strategy: str,
    seed: int,
    ansatz: str,
) -> str:
    """Unique key per container invocation (lane x optimizer x init x seed x ansatz)."""
    return f"{lane}__{optimizer}__{init_strategy}__seed{seed}__{ansatz}"


# ---------------------------------------------------------------------------
# Image verification
# ---------------------------------------------------------------------------

# Maps lane names to the Docker image they require
LANE_IMAGES = {
    "gpu1": "quantum-pipeline:gpu",
    "gpu2": "quantum-pipeline:gpu",
    "cpu":  "quantum-pipeline:cpu",
}

def verify_images(lanes: dict[str, list]) -> None:
    """Verify required Docker images exist locally.

    Images should be built by the upstream build_images DAG task.
    Raises RuntimeError if any image is missing.
    """
    needed = {LANE_IMAGES[lane] for lane in lanes}
    missing = []
    for image_tag in sorted(needed):
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(image_tag)

    if missing:
        raise RuntimeError(
            f"Missing Docker images: {', '.join(missing)}. "
            f"Run the build_images DAG task first."
        )
    logger.info("All required images present: %s", ", ".join(sorted(needed)))


# ---------------------------------------------------------------------------
# Docker command builder
# ---------------------------------------------------------------------------

def build_docker_command(
    service: str,
    molecule_file: str,
    optimizer: str,
    init_strategy: str,
    seed: int,
    ansatz: str,
    config: dict,
    use_gpu: bool,
) -> list[str]:
    """Build the `docker compose run --rm` command for one VQE invocation."""
    compose_path = str(COMPOSE_FILE.relative_to(REPO_ROOT))

    cmd = [
        "docker", "compose",
        "-f", compose_path,
        "--profile", "batch",
        "run", "--rm",
        service,
        "--file", f"./{molecule_file}",
        "--kafka",
        "--basis", config["basis"],
        "--optimizer", optimizer,
        "--init-strategy", init_strategy,
        "--seed", str(seed),
        "--ansatz", ansatz,
        "--simulation-method", config["simulation_method"],
        "--log-level", "INFO",
    ]

    if use_gpu:
        cmd.append("--gpu")

    if config.get("use_convergence"):
        cmd.extend(["--convergence", "--threshold", str(config["threshold"])])

    return cmd


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    run_key: str,
    service: str,
    molecule_file: str,
    optimizer: str,
    init_strategy: str,
    seed: int,
    ansatz: str,
    config: dict,
    use_gpu: bool,
    state_file: Path,
) -> bool:
    """Execute a single VQE lane invocation. Returns True on success."""
    cmd = build_docker_command(
        service, molecule_file, optimizer, init_strategy,
        seed, ansatz, config, use_gpu,
    )

    update_run_state(state_file, run_key, {
        "state": "in_progress",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(cmd),
    })

    logger.info("[%s] Starting: %s | opt=%s init=%s seed=%d ansatz=%s",
                service, run_key.split("__")[0], optimizer, init_strategy, seed, ansatz)
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=7200)
        if result.returncode == 0:
            update_run_state(state_file, run_key, {
                "state": "done",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "return_code": 0,
            })
            logger.info("[%s] Done: %s", service, run_key)
            return True
        stderr_tail = result.stderr[-2000:].decode(errors="replace") if result.stderr else None
        update_run_state(state_file, run_key, {
            "state": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "return_code": result.returncode,
            **({"stderr_tail": stderr_tail} if stderr_tail else {}),
        })
        logger.error("[%s] Failed (rc=%d): %s", service, result.returncode, run_key)
        return False
    except subprocess.TimeoutExpired:
        update_run_state(state_file, run_key, {
            "state": "timed_out",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": "subprocess timed out after 7200s",
        })
        logger.error("[%s] Timed out after 7200s: %s", service, run_key)
        return False
    except Exception as exc:
        update_run_state(state_file, run_key, {
            "state": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        logger.error("[%s] Error: %s - %s", service, run_key, exc)
        return False


# ---------------------------------------------------------------------------
# Matrix generator
# ---------------------------------------------------------------------------

def _phase_sorted_optimizers(optimizers: list[str]) -> list[str]:
    """Return optimizers in phase order (A → B → C), then any unlisted ones."""
    ordered = []
    for phase_opts in OPTIMIZER_PHASES.values():
        for opt in phase_opts:
            if opt in optimizers and opt not in ordered:
                ordered.append(opt)
    for opt in optimizers:
        if opt not in ordered:
            ordered.append(opt)
    return ordered


def generate_run_matrix(
    config: dict,
    lane_molecule_files: dict[str, str],
) -> dict[str, list[dict]]:
    """
    Generate the full experiment matrix partitioned by lane.

    For Tier 3 (multiple ansatz types), iterates over ansatz_types.
    Returns: {lane -> [run_dict, ...]}
    """
    molecule_filter: list[str] | None = config.get("molecule_filter")
    ansatz_types: list[str] = config.get("ansatz_types") or [config["ansatz"]]
    optimizers = _phase_sorted_optimizers(config["optimizers"])

    # Determine which lanes are active (filter out empty lanes for this tier)
    active_lanes: dict[str, str] = {}
    for lane, mol_file in lane_molecule_files.items():
        mol_path = REPO_ROOT / mol_file
        if not mol_path.exists():
            logger.warning("Molecule file not found for lane %s: %s", lane, mol_path)
            continue
        with open(mol_path) as f:
            lane_molecules = [m["name"] for m in json.load(f)]
        if molecule_filter and not any(m in molecule_filter for m in lane_molecules):
            continue  # no molecules in this lane for this tier
        active_lanes[lane] = mol_file

    lanes: dict[str, list[dict]] = {lane: [] for lane in active_lanes}

    for lane, mol_file in active_lanes.items():
        for ansatz in ansatz_types:
            for optimizer in optimizers:
                for init_strategy in config["init_strategies"]:
                    for seed in config["seeds"]:
                        key = make_run_key(lane, optimizer, init_strategy, seed, ansatz)
                        lanes[lane].append({
                            "key": key,
                            "lane": lane,
                            "optimizer": optimizer,
                            "init_strategy": init_strategy,
                            "seed": seed,
                            "ansatz": ansatz,
                            "molecule_file": mol_file,
                            "config": config,
                        })

    return lanes


# ---------------------------------------------------------------------------
# Lane runner
# ---------------------------------------------------------------------------

def run_lane(
    lane: str,
    runs: list[dict],
    state: dict,
    state_file: Path,
    dry_run: bool,
) -> dict[str, int]:
    """Process all runs for a single hardware lane sequentially."""
    service = LANE_SERVICES[lane]
    use_gpu = LANE_GPU[lane]
    results = {"done": 0, "skipped": 0, "failed": 0}

    for run in runs:
        key = run["key"]
        if state.get(key, {}).get("state") == "done":
            results["skipped"] += 1
            continue

        if dry_run:
            cmd = build_docker_command(
                service, run["molecule_file"], run["optimizer"],
                run["init_strategy"], run["seed"], run["ansatz"],
                run["config"], use_gpu,
            )
            logger.info("[DRY RUN][%s] %s", lane, " ".join(cmd))
            results["done"] += 1
            continue

        success = run_experiment(
            run_key=key,
            service=service,
            molecule_file=run["molecule_file"],
            optimizer=run["optimizer"],
            init_strategy=run["init_strategy"],
            seed=run["seed"],
            ansatz=run["ansatz"],
            config=run["config"],
            use_gpu=use_gpu,
            state_file=state_file,
        )
        if success:
            results["done"] += 1
        else:
            results["failed"] += 1

    return results


# ---------------------------------------------------------------------------
# Status reporter
# ---------------------------------------------------------------------------

def _compute_durations(runs: list[dict]) -> list[float]:
    """Parse start/end timestamps from completed runs and return durations in seconds."""
    durations = []
    for r in runs:
        try:
            start = datetime.fromisoformat(r["started_at"])
            end = datetime.fromisoformat(r["completed_at"])
        except (KeyError, ValueError):
            logging.debug('Skipping duration for run with invalid timestamps')
            continue
        durations.append((end - start).total_seconds())
    return durations


def print_status(state_file: Path, lanes: dict[str, list[dict]]) -> None:
    state = load_state(state_file)
    total = sum(len(v) for v in lanes.values())

    counts = {"done": 0, "in_progress": 0, "failed": 0}
    for entry in state.values():
        s = entry.get("state", "pending")
        if s in counts:
            counts[s] += 1
    pending = total - counts["done"] - counts["in_progress"] - counts["failed"]

    print(f"\n{'='*60}")
    print("  VQE Batch Generation Status")
    print(f"{'='*60}")
    print(f"  Total invocations: {total}")
    pct = 100 * counts['done'] // total if total else 0
    print(f"  Done:              {counts['done']} ({pct}%)")
    print(f"  In progress:       {counts['in_progress']}")
    print(f"  Pending:           {pending}")
    print(f"  Failed:            {counts['failed']}")

    print("\n  Per-lane breakdown:")
    for lane, runs in lanes.items():
        lane_keys = {r["key"] for r in runs}
        l_done = sum(1 for k, v in state.items() if k in lane_keys and v.get("state") == "done")
        l_fail = sum(1 for k, v in state.items() if k in lane_keys and v.get("state") == "failed")
        print(f"    {lane:6s}: {l_done}/{len(runs)} done, {l_fail} failed")

    # ETA estimate from recent completed runs
    completed = [
        v for v in state.values()
        if v.get("state") == "done"
        and v.get("started_at")
        and v.get("completed_at")
    ]
    if completed and pending > 0:
        recent = completed[-min(20, len(completed)):]
        durations = _compute_durations(recent)
        if durations:
            avg_sec = sum(durations) / len(durations)
            # 3 lanes running in parallel
            eta_sec = (pending / 3) * avg_sec
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            print(f"\n  ETA: ~{eta_h}h {eta_m}m  (avg {avg_sec:.0f}s/invocation, {pending} pending, 3 lanes parallel)")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQE ML batch generation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=list(TIER_CONFIGS.keys()),
        default=1,
        help="Dataset tier to generate (default: 1)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current generation status and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print docker commands without executing them",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=STATE_FILE,
        help=f"Path to state JSON file (default: {STATE_FILE})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = TIER_CONFIGS[args.tier]
    state_file: Path = args.state_file
    lanes = generate_run_matrix(config, LANE_MOLECULE_FILES)

    if not lanes:
        logger.error("No active lanes for tier %d - check molecule filter and lane files", args.tier)
        return 1

    total = sum(len(v) for v in lanes.values())

    if args.status:
        print_status(state_file, lanes)
        return 0

    state = load_state(state_file)
    done_count = sum(1 for e in state.values() if e.get("state") == "done")
    if done_count > 0:
        logger.info("Resuming: %d/%d invocations already complete", done_count, total)

    logger.info(
        "Tier %d | %d invocations across %d lanes | basis=%s optimizers=%s",
        args.tier, total, len(lanes), config["basis"], config["optimizers"],
    )

    if args.dry_run:
        logger.info("Dry-run mode - no containers will be started")

    if not args.dry_run:
        verify_images(lanes)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            lane: executor.submit(run_lane, lane, runs, state, state_file, args.dry_run)
            for lane, runs in lanes.items()
        }
        all_results = {lane: f.result() for lane, f in futures.items()}

    total_done = sum(r["done"] for r in all_results.values())
    total_skipped = sum(r["skipped"] for r in all_results.values())
    total_failed = sum(r["failed"] for r in all_results.values())

    logger.info(
        "Finished | done=%d skipped=%d failed=%d",
        total_done, total_skipped, total_failed,
    )

    if total_failed > 0:
        logger.warning(
            "%d invocations failed - rerun to resume (completed runs will be skipped)",
            total_failed,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
