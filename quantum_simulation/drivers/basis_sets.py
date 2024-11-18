DEFAULT_BASIS_SETS = ["sto3g", "6-31g", "cc-pVDZ"]


def validate_basis_set(basis_set: str):
    """Check if the basis set is supported."""
    if basis_set not in DEFAULT_BASIS_SETS:
        raise ValueError(
            f"Unsupported basis set: {basis_set}. Choose from {DEFAULT_BASIS_SETS}."
        )
    return basis_set
