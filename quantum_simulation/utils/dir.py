import os
from pathlib import Path

from quantum_simulation.configs import settings


def ensureDirExists(dir_path: str | Path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def buildGraphName(prefix: str, symbols: list[str]):
    graph_name = prefix + '_' + ''.join(str(symbol) for symbol in symbols)
    return graph_name


def savePlot(
    plt,
    path: str | Path,
    prefix: str,
    symbols: list[str],
):
    path = Path(
        ensureDirExists(path),
        buildGraphName(prefix, symbols),
    )
    plt.savefig(path)
    return path.parent / (path.stem + '.png')
