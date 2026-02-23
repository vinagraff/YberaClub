FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "0", "--bind", "0.0.0.0:8080", "main:server"]
