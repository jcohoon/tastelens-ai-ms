# Use slim image with Python 3.10
FROM python:3.10-slim

WORKDIR /app

# Install system-level dependencies for numpy + scikit-surprise
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Pre-install numpy separately to ensure C extensions work
RUN pip install --upgrade pip && \
    pip install numpy

# Copy requirements and install rest of dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]