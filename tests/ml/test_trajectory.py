from quantum_pipeline.configs.settings import SUPPORTED_BASIS_SETS
from quantum_pipeline.ml.trajectory import BASIS_SETS, simulate_descent


class _ScriptedRng:
    """Replays chosen `uniform` draws in call order; noise off, no uphill steps."""

    def __init__(self, uniforms):
        self._uniforms = iter(uniforms)

    def uniform(self, low=0.0, high=1.0):
        return next(self._uniforms)

    def normal(self, loc=0.0, scale=1.0):
        return 0.0

    def random(self):
        return 1.0

    def exponential(self, scale=1.0):
        return scale


def test_stalled_run_descends_when_final_draw_exceeds_start():
    # Draw order for a stalled run: e_final offset, plateau_frac, start fraction. The
    # largest e_final offset against the smallest start fraction puts e_final above the
    # start; the curve must still fall toward the asymptote, not climb to it.
    rng = _ScriptedRng([0.10, 0.4, 0.2])
    molecule = {'e_fci': -1.1175, 'e_local': -0.78}

    descent = simulate_descent(rng, molecule, 50, 'L-BFGS-B', converged=False)

    assert descent.energies[0] > descent.e_final
    assert descent.energies[0] > descent.energies[-1]


def test_basis_sets_are_names_the_pipeline_accepts():
    assert set(BASIS_SETS) <= set(SUPPORTED_BASIS_SETS)
