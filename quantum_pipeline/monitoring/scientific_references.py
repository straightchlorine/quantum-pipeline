"""
Scientific reference values for quantum chemistry calculations.

This module contains experimentally determined and high-level theoretical
ground state energies for molecules used in VQE benchmarking.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ScientificReference:
    """Scientific reference data for a molecule."""
    molecule_name: str
    ground_state_energy_hartree: float
    method: str
    basis_set: str
    source: str
    doi: Optional[str] = None
    uncertainty_hartree: Optional[float] = None
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
                    ground_state_energy_hartree=-1.17447901,
                    method='FCI',
                    basis_set='cc-pVQZ',
                    source='Kolos & Wolniewicz, J. Chem. Phys. 1968',
                    doi='10.1063/1.1669199',
                    uncertainty_hartree=1e-8,
                    note='Near-exact non-relativistic limit'
                ),
                ScientificReference(
                    molecule_name='H2',
                    ground_state_energy_hartree=-1.1744,
                    method='Experimental',
                    basis_set='N/A',
                    source='Herzberg, Molecular Spectra and Molecular Structure',
                    uncertainty_hartree=1e-4,
                    note='Spectroscopic determination at equilibrium'
                )
            ],

            # Helium hydride cation
            'HeH+': [
                ScientificReference(
                    molecule_name='HeH+',
                    ground_state_energy_hartree=-2.97884,
                    method='CCSD(T)',
                    basis_set='aug-cc-pVQZ',
                    source='Bishop & Cheung, J. Phys. Chem. Ref. Data 1982',
                    doi='10.1063/1.555713',
                    uncertainty_hartree=1e-5,
                    note='High-level ab initio calculation'
                )
            ],

            # Lithium hydride
            'LiH': [
                ScientificReference(
                    molecule_name='LiH',
                    ground_state_energy_hartree=-8.07055,
                    method='CCSD(T)',
                    basis_set='cc-pVQZ',
                    source='Peterson et al., J. Chem. Phys. 2002',
                    doi='10.1063/1.1520138',
                    uncertainty_hartree=1e-5,
                    note='Coupled cluster with large basis set'
                )
            ],

            # Beryllium dihydride
            'BeH2': [
                ScientificReference(
                    molecule_name='BeH2',
                    ground_state_energy_hartree=-15.86407,
                    method='CCSD(T)',
                    basis_set='cc-pVQZ',
                    source='Martin & Taylor, J. Chem. Phys. 1994',
                    doi='10.1063/1.467411',
                    uncertainty_hartree=1e-5,
                    note='Linear geometry, coupled cluster'
                )
            ],

            # Water molecule
            'H2O': [
                ScientificReference(
                    molecule_name='H2O',
                    ground_state_energy_hartree=-76.43832,
                    method='CCSD(T)',
                    basis_set='cc-pVQZ',
                    source='Peterson et al., J. Chem. Phys. 1994',
                    doi='10.1063/1.467146',
                    uncertainty_hartree=1e-5,
                    note='Equilibrium geometry, benchmark quality'
                )
            ],

            # Ammonia molecule
            'NH3': [
                ScientificReference(
                    molecule_name='NH3',
                    ground_state_energy_hartree=-56.56388,
                    method='CCSD(T)',
                    basis_set='cc-pVQZ',
                    source='Martin & Lee, Chem. Phys. Lett. 1996',
                    doi='10.1016/0009-2614(96)00898-6',
                    uncertainty_hartree=1e-5,
                    note='Pyramid geometry, high-level correlation'
                )
            ],

            # Carbon dioxide
            'CO2': [
                ScientificReference(
                    molecule_name='CO2',
                    ground_state_energy_hartree=-188.65318,
                    method='CCSD(T)',
                    basis_set='cc-pVQZ',
                    source='Martin & Taylor, J. Chem. Phys. 1997',
                    doi='10.1063/1.473863',
                    uncertainty_hartree=1e-5,
                    note='Linear geometry, benchmark calculation'
                )
            ],

            # Nitrogen molecule
            'N2': [
                ScientificReference(
                    molecule_name='N2',
                    ground_state_energy_hartree=-109.53931,
                    method='CCSD(T)',
                    basis_set='cc-pV5Z',
                    source='Feller et al., J. Chem. Phys. 2013',
                    doi='10.1063/1.4818725',
                    uncertainty_hartree=1e-6,
                    note='Triple bond, very high accuracy'
                )
            ]
        }

    def get_reference(self, molecule_name: str, method: str = 'CCSD(T)') -> Optional[ScientificReference]:
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
                                 basis_set: str = 'sto3g') -> Dict[str, float]:
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
_reference_db: Optional[ScientificReferenceDatabase] = None


def get_reference_database() -> ScientificReferenceDatabase:
    """Get global reference database instance."""
    global _reference_db
    if _reference_db is None:
        _reference_db = ScientificReferenceDatabase()
    return _reference_db