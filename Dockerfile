# Production Dockerfile for 01PS2 – Financial Risk & Fraud Analytics
FROM python:3.11-slim

# Maintainer / Metadata
LABEL maintainer="Financial Engineering Architecture Team"
LABEL project="01PS2-Financial-Risk-Fraud-Analytics"
LABEL version="1.0.0"

# Set non-interactive environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source code
COPY . .

# Generate synthetic dataset and train models
RUN python data/prepare_dataset.py --n-samples 20000 --fraud-ratio 0.01 && \
    python src/train_models.py

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
