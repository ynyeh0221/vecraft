FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install the application
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Launch the Kopf operator
ENTRYPOINT ["kopf", "run", "--standalone", "operator/main.py"]