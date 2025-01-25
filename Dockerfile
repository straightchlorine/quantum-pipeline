FROM python:3.12-slim-bullseye

WORKDIR /usr/src/quantum_pipeline

RUN apt-get update \
      && apt-get install -y git \
      && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir .

CMD [ "python", "quantum_pipeline.py", "-f", "data/molecules.json", "--max-iterations", "100" ]
