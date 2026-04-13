# Multi-stage build
# Stage 1: builder — install deps
FROM cgr.dev/chainguard/python:latest-dev AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM cgr.dev/chainguard/python:latest AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY configs/ configs/
RUN mkdir -p models
COPY .env.example .env
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
