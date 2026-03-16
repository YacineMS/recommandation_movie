FROM apache/airflow:3.1.7-python3.11

WORKDIR /opt/airflow

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /opt/airflow/logs && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 777 /opt/airflow/logs

USER airflow

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
