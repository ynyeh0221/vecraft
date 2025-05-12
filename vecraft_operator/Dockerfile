FROM python:3.11-slim

WORKDIR /app

# Otherwise, install directly
COPY . /app
RUN pip install --no-cache-dir .

# Launch the Kopf operator
ENTRYPOINT ["kopf", "run", "--standalone", "operator/main.py"]
