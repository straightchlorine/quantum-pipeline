FROM apache/airflow:3.1.8

USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk rclone docker.io docker-compose-plugin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars && chown -R airflow:root /tmp/.ivy2
RUN usermod -aG docker airflow || groupadd -g 999 docker && usermod -aG docker airflow

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
