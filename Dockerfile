# Dockerfile

# Use a base image with Python and Conda (or just Python and pip)
# python:3.9-slim-buster is a good choice for smaller images
# If using CUDA, you'd pick a NVIDIA CUDA enabled base image (e.g., nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04)
# For CPU-only, python:3.9-slim-buster is fine.
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
# Install PyTorch specifically for CPU, or follow GPU installation if using CUDA base image
# Use --extra-index-url for PyTorch if needed
RUN pip install --no-cache-dir -r requirements.txt \
    # Install PyTorch for CPU specifically if not using a CUDA image
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    # Additional dependencies for BitsAndBytes (often required)
    && pip install bitsandbytes accelerate

# Set environment variables for model paths inside the container
ENV MODEL_BASE_DIR=/app/models
# Add /app to Python path to ensure imports work
ENV PYTHONPATH=/app 

# Create the models directory
RUN mkdir -p ${MODEL_BASE_DIR}

# Copy the model download and quantization scripts
COPY download_models.py .
COPY quantize_models.py .

# Download and Quantize models during the build process
# This makes the models part of the Docker image
RUN python download_models.py
RUN python quantize_models.py

# Copy the rest of the application code
COPY . .

# Ensure the reports directory exists
RUN mkdir -p /app/analysis_reports

# Expose the port Flask runs on
EXPOSE 5000

# Set environment variables for Flask (optional, but good practice)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Command to run the application using Flask's built-in server
# For production, you would typically use a WSGI server like Gunicorn (e.g., CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"])
CMD ["python", "app.py"]