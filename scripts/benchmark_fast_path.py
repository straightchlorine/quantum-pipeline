"""Benchmark the analytic-gradient fast path against the Aer exact path.

Times one energy+gradient call on the fast path and one energy call on Aer for four molecules
(4, 10, 12 and 14 qubits). The Aer step is estimated as n_params + 1 energy calls, and the
Hartree-Fock point is checked to be stationary. Run from the repository root, one thread per run:
    OMP_NUM_THREADS=1 pdm run python scripts/benchmark_fast_path.py [--output PATH] [--molecules PATH]
It writes benchmarks/fast_path/<date>-<git sha>.json with the per-molecule rows and a
grid_extrapolation in core-hours.
"""

import argparse
import json
import os
import subprocess
import timeit
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import qiskit
import scipy
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, ExcitationPreserving
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from quantum_pipeline import __version__
from quantum_pipeline.circuits import HFData, build_hf_initial_state
from quantum_pipeline.configs.constants import EP_INIT_JITTER
from quantum_pipeline.drivers.molecule_loader import load_molecule, load_molecule_names
from quantum_pipeline.mappers import JordanWignerMapper

# Qubit count after freezing the core and mapping -> (molecule, freeze core).
# build_problem checks the count.
TIERS = {
    4: ('H2', False),
    10: ('LiH', True),
    12: ('BeH2', True),
    14: ('NH3', True),
}

MOLECULES_JSON = 'data/molecules.thesis.json'
MOLECULES_FALLBACK_JSON = 'data/molecules.json'

ANSATZ_REPS = 3
HF_GRAD_TOL = 1e-12
CPU_COUNT = os.cpu_count() or 1

# Core-hour grid: optimizer steps per trajectory, trajectories, and the smallest tier included.
# The 4-qubit tier is overhead-bound, so it stays out.
GRID_STEPS = 300
GRID_TRAJECTORIES = 2600
GRID_MIN_QUBITS = 10
# A 1-minute load average below this share of the CPU count counts as a quiet host.
QUIESCENT_LOAD_SHARE = 0.25

# Built once so its circuit cache is warm, as in the solver.
GRAD = ReverseEstimatorGradient()


def best_of(fn: Callable[[], object], repeat: int = 10) -> float:
    # Minimum, not mean: scheduler noise only adds time. number=1 because one call takes
    # milliseconds.
    return min(timeit.repeat(fn, repeat=repeat, number=1))


def check_hf_stationary(name: str, hf_grad: np.ndarray) -> float:
    norm = float(np.linalg.norm(hf_grad))
    if norm > HF_GRAD_TOL:
        raise RuntimeError(f'{name}: gradient at the Hartree-Fock point is not zero')
    return norm


def energy_and_grad(ansatz: QuantumCircuit, qop: SparsePauliOp, x: np.ndarray) -> tuple[Any, Any]:
    energy = Statevector(ansatz.assign_parameters(x)).expectation_value(qop)
    return energy, GRAD.run([ansatz], [qop], [x]).result().gradients[0]


def build_problem(
    name: str, freeze: bool, expected_qubits: int, molecules: dict[str, MoleculeInfo]
) -> tuple[ElectronicStructureProblem, JordanWignerMapper, SparsePauliOp]:
    problem = PySCFDriver.from_molecule(molecules[name], basis='sto3g').run()
    if freeze:
        problem = FreezeCoreTransformer().transform(problem)
    mapper = JordanWignerMapper()
    qop = mapper.map(problem.second_q_ops()[0])
    if qop.num_qubits != expected_qubits:
        raise RuntimeError(f'{name}: expected {expected_qubits} qubits, got {qop.num_qubits}')
    return problem, mapper, qop


def time_aer(ansatz: QuantumCircuit, qop: SparsePauliOp, x: np.ndarray) -> float:
    # Transpiled as the solver does, with the CLI default optimization_level=3.
    # Only the estimator call is timed.
    backend = AerSimulator(method='statevector')
    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=3, seed_transpiler=42
    )
    isa = pm.run(ansatz)
    hisa = qop.apply_layout(isa.layout)
    est = AerEstimatorV2.from_backend(backend, options={'default_precision': 0.0})
    return best_of(lambda: est.run([(isa, [hisa], [x])]).result()[0].data.evs[0])


def get_git_sha() -> str:
    # git may be absent (GPU host image): fall back to QP_GIT_SHA, then 'unknown'.
    try:
        sha = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = ''
    return sha or os.environ.get('QP_GIT_SHA') or 'unknown'


def resolve_molecules_path(path: Path) -> Path:
    # Only the default may fall back; an explicit --molecules path is used as given.
    fallback = Path(MOLECULES_FALLBACK_JSON)
    if str(path) == MOLECULES_JSON and not path.exists() and fallback.exists():
        return fallback
    return path


def run_metadata() -> dict[str, Any]:
    return {
        'git_sha': get_git_sha(),
        'package_version': __version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'qiskit': qiskit.__version__,
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'cpu_count': CPU_COUNT,
    }


def measure_tier(
    name: str,
    freeze: bool,
    expected_qubits: int,
    molecules: dict[str, MoleculeInfo],
    meta: dict[str, Any],
) -> dict[str, Any]:
    problem, mapper, qop = build_problem(name, freeze, expected_qubits, molecules)

    ansatz = EfficientSU2(qop.num_qubits, reps=ANSATZ_REPS)
    # One fixed parameter vector per molecule, so timings do not depend on a lucky draw.
    x = 2 * np.pi * np.random.default_rng(42).random(ansatz.num_parameters)
    t_energy_s = best_of(lambda: Statevector(ansatz.assign_parameters(x)).expectation_value(qop))
    t_energy_grad_s = best_of(lambda: energy_and_grad(ansatz, qop, x))
    t_aer_eval_s = time_aer(ansatz, qop, x)
    # One exact-path optimizer step: one energy call per parameter plus the base point.
    aer_step_s = t_aer_eval_s * (ansatz.num_parameters + 1)

    # ExcitationPreserving as built in VQESolver._build_ansatz. Its XX+YY gates are translated
    # before differentiation, so one call costs more than for EfficientSU2.
    hf_circuit = build_hf_initial_state(
        HFData(
            num_particles=problem.num_particles,
            num_spatial_orbitals=problem.num_spatial_orbitals,
        ),
        mapper,
    )
    ep_ansatz = ExcitationPreserving(
        qop.num_qubits, reps=ANSATZ_REPS, entanglement='full', initial_state=hf_circuit
    )
    hf_grad = (
        GRAD.run([ep_ansatz], [qop], [np.zeros(ep_ansatz.num_parameters)]).result().gradients[0]
    )
    hf_grad_norm = check_hf_stationary(name, hf_grad)

    # Timed at the solver's default start: EP_INIT_JITTER * N(0, 1) noise.
    x_ep = EP_INIT_JITTER * np.random.default_rng(42).standard_normal(ep_ansatz.num_parameters)
    t_energy_grad_ep_s = best_of(lambda: energy_and_grad(ep_ansatz, qop, x_ep))

    return {
        'molecule_name': name,
        'geometry': [list(c) for c in molecules[name].coords],
        'n_qubits': qop.num_qubits,
        'n_params': ansatz.num_parameters,
        'n_params_ep': ep_ansatz.num_parameters,
        't_energy_s': t_energy_s,
        't_energy_grad_s': t_energy_grad_s,
        't_energy_grad_ep_s': t_energy_grad_ep_s,
        't_aer_eval_s': t_aer_eval_s,
        'aer_step_s': aer_step_s,
        'speedup': aer_step_s / t_energy_grad_s,
        'hf_grad_norm': hf_grad_norm,
        'frozen_core': freeze,
        **meta,
        'loadavg': os.getloadavg(),
    }


def grid_extrapolation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Core-hours for the full grid, from the measured per-step times of the 10-14 qubit tiers.

    One thread per run, so wall seconds equal core seconds.
    """
    cells = [r for r in rows if r['n_qubits'] >= GRID_MIN_QUBITS]
    per_step = {
        'efficientsu2': [r['t_energy_grad_s'] for r in cells],
        'excitation_preserving': [r['t_energy_grad_ep_s'] for r in cells],
        'aer_exact': [r['aer_step_s'] for r in cells],
    }
    levels = {}
    for key, times in per_step.items():
        cell_s = [GRID_STEPS * t for t in times]
        core_h = [GRID_TRAJECTORIES * c / 3600 for c in cell_s]
        levels[key] = {
            'step_s_range': [min(times), max(times)],
            'cell_s_range': [min(cell_s), max(cell_s)],
            'core_hours_range': [min(core_h), max(core_h)],
        }
    load1 = max(r['loadavg'][0] for r in rows)
    return {
        'steps_per_trajectory': GRID_STEPS,
        'trajectories': GRID_TRAJECTORIES,
        'grid_qubits': [r['n_qubits'] for r in cells],
        'levels': levels,
        'max_loadavg_1min': load1,
        'quiescent_host': load1 < QUIESCENT_LOAD_SHARE * CPU_COUNT,
    }


def print_summary(
    rows: list[dict[str, Any]], extrapolation: dict[str, Any], out_path: Path
) -> None:
    print('qubits  params(su2/ep)  fast_s   ep_s     aer_eval_s  aer_step_s  speedup  hf_grad')
    for r in rows:
        print(
            f'{r["molecule_name"]:<5} {r["n_qubits"]:>3}q  {r["n_params"]:>4}/{r["n_params_ep"]:<4}  '
            f'{r["t_energy_grad_s"]:.4f}  {r["t_energy_grad_ep_s"]:.4f}  {r["t_aer_eval_s"]:.4f}  '
            f'{r["aer_step_s"]:9.2f}  {r["speedup"]:7.1f}x  {r["hf_grad_norm"]:.1e}'
        )
    for key, v in extrapolation['levels'].items():
        lo, hi = v['core_hours_range']
        print(f'grid {key}: {lo:.0f}-{hi:.0f} core-hours')
    print(f'wrote {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, default=None, help='result JSON path')
    parser.add_argument(
        '--molecules', type=Path, default=Path(MOLECULES_JSON), help='molecules JSON path'
    )
    args = parser.parse_args()

    mol_path = str(resolve_molecules_path(args.molecules))
    molecules = dict(zip(load_molecule_names(mol_path), load_molecule(mol_path), strict=True))
    meta = run_metadata()
    rows = [measure_tier(name, freeze, n, molecules, meta) for n, (name, freeze) in TIERS.items()]
    extrapolation = grid_extrapolation(rows)

    # <date>-<sha>.json: a rerun on the same day and commit overwrites the earlier file.
    out_path = args.output or (
        Path('benchmarks/fast_path') / f'{date.today():%Y-%m-%d}-{meta["git_sha"]}.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'rows': rows, 'grid_extrapolation': extrapolation}
    out_path.write_text(json.dumps(payload, indent=2) + '\n')
    print_summary(rows, extrapolation, out_path)


if __name__ == '__main__':
    main()
