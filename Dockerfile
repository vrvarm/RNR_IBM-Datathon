FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Build tools are a fallback for architectures (e.g. s390x on LinuxONE) where a package has no prebuilt wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fraud ./fraud
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["scripts/start.sh"]
