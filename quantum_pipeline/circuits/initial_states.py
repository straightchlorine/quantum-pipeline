"""
initial_states.py

Hartree-Fock initial state preparation for VQE ansatz.

Provides a HartreeFock circuit (from qiskit-nature) to be used as the
initial_state parameter of EfficientSU2. This correctly encodes the HF
ground state under the given qubit mapping with blocked spin-orbital ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock

from quantum_pipeline.mappers.mapper import Mapper


@dataclass
class HFData:
    """Hartree-Fock data extracted from a PySCF electronic structure problem."""

    num_particles: tuple[int, int]
    num_spatial_orbitals: int
    reference_energy: float | None = None


def build_hf_initial_state(hf_data: HFData, mapper: Mapper) -> QuantumCircuit:
    """Build a HartreeFock initial state circuit for EfficientSU2.

    Uses qiskit-nature's HartreeFock circuit which correctly handles
    blocked spin-orbital ordering (alpha block, then beta block).

    Args:
        hf_data: Hartree-Fock data with particle counts and spatial orbitals.
        mapper: A Mapper instance whose get_qiskit_mapper() provides the
                qiskit-nature qubit mapper.

    Returns:
        QuantumCircuit preparing the exact HF state.
    """
    return HartreeFock(
        num_spatial_orbitals=hf_data.num_spatial_orbitals,
        num_particles=hf_data.num_particles,
        qubit_mapper=mapper.get_qiskit_mapper(),
    )
