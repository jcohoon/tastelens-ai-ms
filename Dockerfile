# Use slim image with Python 3.10
FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ gfortran libatlas-base-dev && \
    pip install --upgrade pip && \
    pip install numpy

# Copy and install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]