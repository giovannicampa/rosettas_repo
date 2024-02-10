# Assuming you're using an NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# # Install the project dependencies via setup.py
RUN pip install -e .
