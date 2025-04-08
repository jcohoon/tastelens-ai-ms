# Use slim image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for scikit-surprise
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev gfortran && \
    pip install --upgrade pip && \
    pip install numpy  # numpy first, to prevent surprise build issues

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]