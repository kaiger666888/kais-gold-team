FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy v6 module
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY engines/ ./engines/

RUN groupadd -r kais && useradd -r -g kais -m -d /home/kais kais && \
    mkdir -p /home/kais/.cache && \
    chown -R kais:kais /app /home/kais
USER kais

ENV PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "src.v6.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
