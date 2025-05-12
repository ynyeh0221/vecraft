FROM python:3.11-slim

# Install build dependencies for C++ extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package code
COPY . /app

# Install dependencies with special handling for hnswlib
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall --no-deps hnswlib && \
    pip install --no-cache-dir .

ENTRYPOINT ["true"]