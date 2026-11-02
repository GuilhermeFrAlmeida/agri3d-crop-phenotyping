# Development/baseline image. Production API belongs to a later phase.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir '.[geometry]'
ENTRYPOINT ["python", "scripts/evaluate_baseline.py"]
CMD ["--help"]
