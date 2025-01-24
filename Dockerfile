FROM python:3.12-slim-bullseye

WORKDIR /usr/src/quantum-data-generator

RUN apt-get update \
      && apt-get install -y git \
      && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

CMD [ "python", "quantum_pipeline.py", "-f", "data/molecules.json", "--max-iterations", "1", "--kafka" ]
