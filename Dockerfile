FROM python:3.12-slim-bullseye

WORKDIR /usr/src/quantum-data-generator

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    numpy \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "main.py", "-f", "data/molecules.json" ]
