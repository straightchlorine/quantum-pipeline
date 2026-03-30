FROM apache/airflow:3.1.8

# Host docker socket GID - override if your system differs (check: stat -c '%g' /var/run/docker.sock)
ARG DOCKER_GID=970

USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk rclone docker.io docker-compose-plugin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars && chown -R airflow:root /tmp/.ivy2

# Match the host's docker socket GID so airflow can use the mounted socket
RUN groupadd -g ${DOCKER_GID} hostdocker 2>/dev/null || true && \
    usermod -aG ${DOCKER_GID} airflow

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
