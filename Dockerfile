FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-numpy python3-pandas \
    python3-scipy python3-sklearn python3-joblib curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY fraud ./fraud
COPY scripts ./scripts
RUN chmod +x scripts/*.sh
EXPOSE 8000
CMD ["scripts/start.sh"]
