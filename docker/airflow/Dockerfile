FROM apache/airflow:3.1.8

USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars && chown -R airflow:root /tmp/.ivy2

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
