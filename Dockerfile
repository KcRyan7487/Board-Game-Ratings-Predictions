FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files needed for inference web app
COPY app.py ./
COPY inference_utils.py ./
COPY templates ./templates
COPY artifacts ./artifacts

EXPOSE 5000

# Heroku-style PORT support + local fallback
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app"]

