"""Tests for scripts/generate_ml_batch.py run-key construction.

Regression for the HIGH finding: make_run_key omitted the basis set, so tiers
that differ only by basis (Tier 1 sto3g vs Tier 2 6-31g vs Tier 4 cc-pvdz)
collided on the same key. Once one tier marked a key 'done', the other was
silently skipped while the batch reported 100% complete.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from generate_ml_batch import build_docker_command, make_run_key

_BASE = {
    'lane': 'cpu',
    'optimizer': 'L-BFGS-B',
    'init_strategy': 'random',
    'seed': 1,
    'ansatz': 'EfficientSU2',
    'molecule': 'H2',
}


def test_run_key_includes_basis():
    """Same run config across different basis sets must yield distinct keys."""
    sto3g = make_run_key(basis='sto3g', **_BASE)
    six31g = make_run_key(basis='6-31g', **_BASE)
    ccpvdz = make_run_key(basis='cc-pvdz', **_BASE)
    assert sto3g != six31g != ccpvdz
    assert len({sto3g, six31g, ccpvdz}) == 3
    assert 'sto3g' in sto3g and '6-31g' in six31g and 'cc-pvdz' in ccpvdz


def test_run_key_is_deterministic():
    """Identical arguments produce an identical key (idempotent resume relies on this)."""
    assert make_run_key(basis='sto3g', **_BASE) == make_run_key(basis='sto3g', **_BASE)


def test_run_key_distinguishes_every_dimension():
    """Changing any single dimension changes the key (no accidental collisions)."""
    base = make_run_key(basis='sto3g', **_BASE)
    variants = {
        make_run_key(basis='sto3g', **{**_BASE, 'lane': 'gpu1'}),
        make_run_key(basis='sto3g', **{**_BASE, 'optimizer': 'COBYLA'}),
        make_run_key(basis='sto3g', **{**_BASE, 'init_strategy': 'hf'}),
        make_run_key(basis='sto3g', **{**_BASE, 'seed': 2}),
        make_run_key(basis='sto3g', **{**_BASE, 'ansatz': 'RealAmplitudes'}),
        make_run_key(basis='sto3g', **{**_BASE, 'molecule': 'LiH'}),
        make_run_key(basis='6-31g', **_BASE),
    }
    assert base not in variants
    assert len(variants) == 7  # all distinct


_CONFIG = {'basis': 'sto3g', 'simulation_method': 'statevector'}


def _command(optimizer):
    return build_docker_command(
        'quantum-pipeline-cpu', 'H2.json', optimizer, 'random', 1, 'EfficientSU2', _CONFIG, False
    )


def test_command_passes_exact_for_gradient_optimizer():
    """Sampled shots with a gradient optimizer are rejected, so the batch must pass --exact."""
    assert '--exact' in _command('L-BFGS-B')


def test_command_passes_exact_for_every_optimizer():
    """The batch builds exact trajectories for all optimizers, COBYLA included."""
    assert '--exact' in _command('COBYLA')
