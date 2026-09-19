from collections.abc import Sequence
from pathlib import Path


def save_plot(
    plt,
    path: str | Path,
    prefix: str,
    symbols: list[str],
) -> Path:
    path = Path(
        ensure_dir_exists(path),
        build_graph_name(prefix, symbols),
    )
    plt.savefig(path)
    return path.parent / (path.stem + '.png')


def ensure_dir_exists(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_graph_name(prefix: str, symbols: list[str] | Sequence[str]) -> str:
    return f'{prefix}_{"_".join(symbols)}'


def get_graph_path(path: str | Path, prefix: str, symbols: list[str] | Sequence[str]) -> Path:
    """Generate unique file path for graph, appending counter suffix if file exists."""
    dir_path = ensure_dir_exists(path)
    base_name = build_graph_name(prefix, symbols)
    file_path = Path(dir_path) / f'{base_name}.png'

    counter = 0
    while file_path.exists():
        counter += 1
        unique_suffix = f'_{counter}'
        file_path = Path(dir_path) / f'{base_name}{unique_suffix}.png'

    return file_path
