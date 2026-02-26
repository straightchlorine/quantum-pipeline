"""
Scientific reference values for quantum chemistry calculations.

This module contains experimentally determined and high-level theoretical
ground state energies for molecules used in VQE benchmarking.
"""

from dataclasses import dataclass


@dataclass
class ScientificReference:
    """Scientific reference data for a molecule."""
    molecule_name: str
    ground_state_energy_hartree: float
    method: str
    basis_set: str
    source: str
    doi: str | None = None
    uncertainty_hartree: float | None = None
    note: str = ""


class ScientificReferenceDatabase:
    """Database of high-accuracy reference values for VQE benchmarking."""

    def __init__(self):
        """Initialize the reference database with literature values."""
        self.references = {
            # Hydrogen molecule - experimental and high-level theory
            'H2': [
                ScientificReference(
                    molecule_name='H2',
                    ground_state_energy_hartree=-1.164025030884,
                    method='Nonrelativistic Schrödinger',
                    basis_set='naJC exponentials',
                    source='Pachucki & Komasa, J. Chem. Phys. 2016',
                    doi='10.1063/1.4948309',
                    uncertainty_hartree=1e-12,
                    note='Unprecedented accuracy 10⁻¹². D₀ = 36118.7977463(2) cm⁻¹. Table I, p. 2'
                ),
                ScientificReference(
                    molecule_name='H2',
                    ground_state_energy_hartree=-1.117,
                    method='HF',
                    basis_set='sto-3g',
                    source='Szabo & Ostlund, Modern Quantum Chemistry, p. 108; Pachucki & Komasa, J. Chem. Phys. 2016, Table I, p. 2',
                    note='Hartree-Fock with STO-3G basis set'
                )
            ],

            # Helium hydride cation
            'HeH+': [
                ScientificReference(
                    molecule_name='HeH+',
                    ground_state_energy_hartree=-2.927,
                    method='Full CI',
                    basis_set='sto-3g',
                    source='Szabo & Ostlund, Modern Quantum Chemistry, p. 231; Table 3.6, p. 178',
                    note='Full Configuration Interaction with STO-3G basis set'
                )
            ],

            # Lithium hydride
            'LiH': [
                ScientificReference(
                    molecule_name='LiH',
                    ground_state_energy_hartree=-7.882,
                    method='CCSD',
                    basis_set='sto-3g',
                    source='Szabo & Ostlund, Modern Quantum Chemistry, p. 286; Avramidis et al., arXiv:2401.17054, Table I, p. 4',
                    note='Coupled Cluster Singles and Doubles with STO-3G. VQE with UCCSD ansatz in Qiskit simulator'
                )
            ],

            # Beryllium dihydride
            'BeH2': [
                ScientificReference(
                    molecule_name='BeH2',
                    ground_state_energy_hartree=-15.56089,
                    method='FCI',
                    basis_set='CAS(2,3)',
                    source='Belaloui et al., arXiv:2412.02606, 2024',
                    doi='10.48550/arXiv.2412.02606',
                    uncertainty_hartree=1e-5,
                    note='FCI with CAS(2,3) approximation, Be-H bond length 1.326 Å. VQE on IBM Fez QPU achieved -15.55901 Ha'
                )
            ],

            # Water molecule
            'H2O': [
                ScientificReference(
                    molecule_name='H2O',
                    ground_state_energy_hartree=-74.963,
                    method='HF',
                    basis_set='sto-3g',
                    source='Szabo & Ostlund, Modern Quantum Chemistry, p. 108; Table 3.13, p. 192',
                    note='Hartree-Fock with STO-3G for ten-electron series (CH4, NH3, H2O, FH)'
                )
            ],

            # Ammonia molecule
            'NH3': [
                ScientificReference(
                    molecule_name='NH3',
                    ground_state_energy_hartree=-55.454,
                    method='HF',
                    basis_set='sto-3g',
                    source='Szabo & Ostlund, Modern Quantum Chemistry, p. 108; Table 3.13, p. 192',
                    note='Hartree-Fock with STO-3G for ten-electron series (CH4, NH3, H2O, FH)'
                )
            ],

        }

    def get_reference(self, molecule_name: str, method: str = 'CCSD(T)') -> ScientificReference | None:
        """
        Get reference value for a molecule.

        Args:
            molecule_name: Name of molecule (e.g., 'H2', 'H2O')
            method: Preferred method ('CCSD(T)', 'FCI', 'Experimental')

        Returns:
            ScientificReference object or None if not found
        """
        molecule_refs = self.references.get(molecule_name, [])

        # First try to find exact method match
        for ref in molecule_refs:
            if ref.method == method:
                return ref

        # Fall back to first available reference
        if molecule_refs:
            return molecule_refs[0]

        return None

    def calculate_accuracy_metrics(self, molecule_name: str, vqe_energy: float,
                                 basis_set: str = 'sto3g') -> dict[str, float]:
        """
        Calculate accuracy metrics for VQE result vs reference.

        Args:
            molecule_name: Name of molecule
            vqe_energy: VQE calculated energy in Hartree
            basis_set: Basis set used in VQE calculation

        Returns:
            Dictionary with accuracy metrics
        """
        reference = self.get_reference(molecule_name)
        if not reference:
            return {
                'reference_available': False,
                'reference_energy_hartree': None,
                'energy_error_hartree': None,
                'energy_error_millihartree': None,
                'relative_error_percent': None,
                'accuracy_score': None
            }

        # Calculate errors
        energy_error = vqe_energy - reference.ground_state_energy_hartree
        relative_error = abs(energy_error / reference.ground_state_energy_hartree) * 100

        # Accuracy score (0-100, where 100 is perfect)
        # Based on chemical accuracy (1 mH = 0.001 Hartree)
        chemical_accuracy_threshold = 0.001  # 1 millihartree
        accuracy_score = max(0, 100 * (1 - abs(energy_error) / chemical_accuracy_threshold))

        return {
            'reference_available': True,
            'reference_energy_hartree': reference.ground_state_energy_hartree,
            'reference_method': reference.method,
            'reference_source': reference.source,
            'energy_error_hartree': energy_error,
            'energy_error_millihartree': energy_error * 1000,
            'relative_error_percent': relative_error,
            'accuracy_score': min(100, accuracy_score),
            'within_chemical_accuracy': abs(energy_error) <= chemical_accuracy_threshold,
            'basis_set_correction_needed': basis_set.lower() in ['sto3g', 'sto-3g']
        }

    def get_all_molecules(self) -> list[str]:
        """Get list of all molecules with reference data."""
        return list(self.references.keys())

    def get_reference_summary(self, molecule_name: str) -> str:
        """Get formatted summary of reference data for a molecule."""
        refs = self.references.get(molecule_name, [])
        if not refs:
            return f"No reference data available for {molecule_name}"

        summary = f"Reference data for {molecule_name}:\n"
        for i, ref in enumerate(refs, 1):
            summary += f"  {i}. {ref.method}/{ref.basis_set}: {ref.ground_state_energy_hartree:.6f} Ha\n"
            summary += f"     Source: {ref.source}\n"
            if ref.note:
                summary += f"     Note: {ref.note}\n"

        return summary


# Global instance for easy access
_reference_db: ScientificReferenceDatabase | None = None


def get_reference_database() -> ScientificReferenceDatabase:
    """Get global reference database instance."""
    global _reference_db
    if _reference_db is None:
        _reference_db = ScientificReferenceDatabase()
    return _reference_db
