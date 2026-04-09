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

# Expose the port the API runs on
EXPOSE 8001

# --- Command ---
CMD ["uvicorn", "src.otwin8_api:app", "--host", "0.0.0.0", "--port", "8001"] 