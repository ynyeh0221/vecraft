# To enable HTTP/2 support, consider one of the following approaches:
#
# Option 1: Use uvicorn directly with HTTP/2 support (requires HTTPS)
# - Install: RUN pip install --no-cache-dir uvicorn[standard]
# - Add SSL certificate files: cert.pem and key.pem inside a certs/ directory
# - Update ENTRYPOINT and CMD:
#     ENTRYPOINT ["uvicorn", "vecraft_db.rest.vecraft_rest_api_server:app"]
#     CMD ["--host", "0.0.0.0", "--port", "8080", "--http", "h2", "--ssl-keyfile", "certs/key.pem", "--ssl-certfile", "certs/cert.pem"]
#
# Option 2: Use Hypercorn as the ASGI server (supports HTTP/2 and HTTP/3)
# - Install: RUN pip install --no-cache-dir hypercorn
# - Add SSL certificate files: cert.pem and key.pem inside a certs/ directory
# - Update ENTRYPOINT and CMD:
#     ENTRYPOINT ["hypercorn", "vecraft_db.rest.vecraft_rest_api_server:app"]
#     CMD ["--bind", "0.0.0.0:8080", "--certfile", "certs/cert.pem", "--keyfile", "certs/key.pem", "--alpn-protocols", "h2"]
#
# Note: HTTP/2 generally requires TLS (HTTPS). Most browsers won't use HTTP/2 over plain HTTP.
# For development, you can generate a self-signed certificate:
#     openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

# 1. Use official Python runtime as parent image
FROM python:3.11-slim

# 2. Set application directory (avoids accidental root context)
WORKDIR /app

# 3. Install build dependencies for C++ extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only metadata to leverage Docker caching
COPY pyproject.toml README.md ./

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
       gunicorn~=21.2.0 \
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

# 9. Define entrypoint and default CMD to allow argument overrides
ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "vecraft_db.rest.vecraft_rest_api_server:app"]
CMD ["--bind", "0.0.0.0:8080", "--workers", "4"]