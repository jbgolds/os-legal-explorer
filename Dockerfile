FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    gcc \
    g++ \
    python3-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Mount the code as a volume instead of copying
# (we'll mount the code from the host in docker-compose)

# Install project in development mode
COPY pyproject.toml .
RUN pip install -e .

# Copy source code first
COPY src/ ./src/

# Install npm dependencies and build CSS
COPY package.json ./
COPY tailwind.config.js ./
RUN npm install && npm run build:css


# Basic environment settings
ENV PYTHONPATH=/app
ENV PORT=8000
# Don't write .pyc files (cleaner)
ENV PYTHONDONTWRITEBYTECODE=1
# Don't buffer output (better logging)
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE ${PORT}

# Use entrypoint script
# ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Simple development command
# "--proxy-headers", for when running behind caddy
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--reload"] 