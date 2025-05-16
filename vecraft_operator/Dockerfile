# 1. Use official Python runtime as parent image
FROM python:3.12-slim

# 2. Set application directory (avoids accidental root context)
WORKDIR /app

# 3. Install build tools for any C/C++ extensions (if needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only metadata to leverage Docker caching
COPY pyproject.toml README.md ./

# 5. Upgrade pip and install all Python dependencies in a single layer,
#    sorted alphabetically, no extra layers
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       kopf~=1.38.0 \
       kubernetes~=24.2.0

# 6. Copy only your Python package and packaging metadata
#    (everything else—charts, yamls, docs—should be in .dockerignore)
COPY vecraft_operator/operator/           ./operator
COPY pyproject.toml      ./

# 7. (same as before)
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# 8. Install your operator package itself without re-installing deps
RUN pip install --no-cache-dir -e . --no-dependencies

# 9. Default command
ENTRYPOINT ["kopf", "run", "/operator/operator.py", "--verbose"]
