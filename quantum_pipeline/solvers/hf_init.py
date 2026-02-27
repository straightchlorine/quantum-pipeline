"""
hf_init.py

Hartree-Fock based initial parameter computation for EfficientSU2 ansatz.

Under Jordan-Wigner mapping, the HF state occupies the first n_alpha + n_beta
qubits. For EfficientSU2 (Ry/Rz gates):
- Layer 0 Ry block: Ry=π for occupied qubits (flips |0⟩→|1⟩), Ry=0 for virtual
- All Rz and subsequent layers: 0 (identity)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HFData:
    """Hartree-Fock data extracted from a PySCF electronic structure problem."""

    num_particles: tuple[int, int]
    num_spatial_orbitals: int
    reference_energy: float | None = None


def compute_hf_initial_parameters(
    n_qubits: int,
    hf_data: HFData,
    ansatz_reps: int,
) -> np.ndarray:
    """Compute EfficientSU2 initial parameters from Hartree-Fock state.

    EfficientSU2 parameter layout per layer: [Ry_0..Ry_{n-1}, Rz_0..Rz_{n-1}]
    Total parameters: n_qubits * 2 * (ansatz_reps + 1)

    Args:
        n_qubits: Number of qubits in the circuit.
        hf_data: Hartree-Fock data with particle counts.
        ansatz_reps: Number of repetition layers in EfficientSU2.

    Returns:
        np.ndarray of initial parameters.

    Raises:
        ValueError: If occupied orbitals exceed available qubits.
    """
    n_alpha, n_beta = hf_data.num_particles
    n_occupied = n_alpha + n_beta

    if n_occupied > n_qubits:
        raise ValueError(
            f'Number of occupied orbitals ({n_occupied}) exceeds '
            f'number of qubits ({n_qubits})'
        )

    num_layers = ansatz_reps + 1
    params_per_layer = n_qubits * 2  # Ry block + Rz block
    total_params = params_per_layer * num_layers

    params = np.zeros(total_params)

    # Layer 0, Ry block: set π for occupied qubits
    for i in range(n_occupied):
        params[i] = np.pi

    return params
