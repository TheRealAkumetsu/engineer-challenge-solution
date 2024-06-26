# Use the Miniforge3 image as the base for ARM64
FROM condaforge/miniforge3

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the model file into the container
COPY weights_final.h5 /app/weights_final.h5

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Flask and transformers via conda
RUN conda install -c conda-forge flask transformers

# Install TensorFlow and pytest via pip
RUN pip install tensorflow==2.10.0 pytest

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]