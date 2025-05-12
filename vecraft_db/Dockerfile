FROM python:3.11-slim

WORKDIR /app

# Copy package code
COPY . /app

# Install the DB core library
RUN pip install --no-cache-dir .

ENTRYPOINT ["true"]
