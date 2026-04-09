# --- Base Image ---
FROM python:3.11-slim

# --- Environment Variables ---
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Install Python Dependencies ---
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy configuration files
COPY config/ ./config/

ENV PORT=5010

# Expose the port the API runs on
EXPOSE ${PORT}

# --- Command ---
CMD ["sh", "-c", "uvicorn src.otwin8_api:app --host 0.0.0.0 --port ${PORT}"]