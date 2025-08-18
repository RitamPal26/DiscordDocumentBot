FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Stage 2: Runtime - Slim image for production
# AND CHANGE THIS LINE
FROM python:3.11-slim

WORKDIR /app
# Update the python version in the path below to match the version above (e.g., python3.11)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
CMD ["python", "bot.py"]