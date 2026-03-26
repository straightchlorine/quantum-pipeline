import sys

sys.argv[0] = 'quantum-pipeline'

from quantum_pipeline.cli import main  # noqa: E402

main()
