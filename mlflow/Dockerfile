FROM python:3.11-bullseye

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["mlflow", "ui", "--backend-store-uri=sqlite:///mlflow.db", "--default-artifact-root=/app/artifacts","--host=0.0.0.0","--port=5000","--allowed-hosts=*"]