# 1. Use official Python runtime as parent image
FROM python:3.11-slim

# 2. Set application directory (avoids accidental root context)
WORKDIR /app

# 3. Install build dependencies for C++ extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only metadata to leverage Docker caching
COPY pyproject.toml setup.py setup.cfg README.md ./

# 5. Upgrade pip and install all Python dependencies in a single RUN,
#    with packages sorted alphanumerically and no extra layers
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --force-reinstall --no-deps hnswlib \
    && pip install --no-cache-dir \
       fastapi~=0.110.0 \
       httpx~=0.27.0 \
       matplotlib~=3.9.4 \
       numpy~=2.0.2 \
       prometheus-client~=0.20.0 \
       pydantic~=2.11.3 \
       pyyaml~=6.0.1 \
       scikit-learn~=1.6.1 \
       scipy \
       typer~=0.15.2

# 6. Copy application source code
#    Ensure sensitive files (e.g., .env, credentials, .git) are excluded via .dockerignore
COPY . .

# 7. Create and switch to a non-root user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# 8. Install the application itself without touching already-installed dependencies
RUN pip install --no-cache-dir -e . --no-dependencies

# 9. Default command (override as needed)
ENTRYPOINT ["true"]
