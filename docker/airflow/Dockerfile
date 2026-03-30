FROM apache/airflow:3.1.8

USER root

# Docker CE CLI + buildx + compose plugin (from Docker's apt repo)
RUN apt-get update && \
    apt-get install -y ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y openjdk-17-jdk rclone docker-ce-cli docker-buildx-plugin docker-compose-plugin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/.ivy2/cache /tmp/.ivy2/jars && chown -R airflow:root /tmp/.ivy2
RUN groupadd -g 999 docker 2>/dev/null || true && usermod -aG docker airflow

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
