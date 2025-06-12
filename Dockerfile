ARG BASE=nvidia/cuda:12.2.0-base-ubuntu22.04
FROM ${BASE}

# Install OS-level dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ make \
    python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev festival ffmpeg git hdf5-tools mbrola && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python3 -m pip install --upgrade pip
# Install llvmlite separately due to build quirks
RUN pip3 install llvmlite --ignore-installed
# Pre-install torch & torchaudio (CUDA-accelerated)
RUN pip3 install torch==2.4.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# Copy requirements file early to leverage Docker cache
COPY requirements.txt .
# Pre-install Python dependencies to benefit from layer caching
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app
WORKDIR /app

# Install editable package (e.g., local modules)
RUN pip install -e .[all,dev,notebooks]
# Clean up to reduce image size
RUN rm -rf /root/.cache ~/.cache

# Copy TTS repository contents:
WORKDIR /app

# Set the default command to execute main.py
CMD ["python3", "processing/inference.py"]
