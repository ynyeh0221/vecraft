FROM python:3.11-slim

# Install build dependencies for C++ extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY pyproject.toml setup.py setup.cfg README.md ./
# If you have a requirements.txt file, copy it here too
# COPY requirements.txt ./

# Handle hnswlib separately first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall --no-deps hnswlib

# Install dependencies in smaller groups to avoid timeout/cancellation
# Group 1: Core numeric and scientific libraries
RUN pip install --no-cache-dir numpy~=2.0.2 scipy

# Group 2: ML and visualization
RUN pip install --no-cache-dir scikit-learn~=1.6.1 matplotlib~=3.9.4

# Group 3: API and utilities
RUN pip install --no-cache-dir pydantic~=2.11.3 typer~=0.15.2 fastapi~=0.110.0 httpx~=0.27.0

# Group 4: Remaining dependencies
RUN pip install --no-cache-dir pyyaml~=6.0.1 prometheus-client~=0.20.0

# Now copy the rest of the code
COPY . .

# Finally install the package itself, without reinstalling dependencies
RUN pip install --no-cache-dir -e . --no-dependencies

# Set default command
ENTRYPOINT ["true"]