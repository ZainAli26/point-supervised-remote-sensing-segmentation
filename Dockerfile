FROM nvidia/cuda:12.5.0-base-ubuntu22.04

WORKDIR /app

# Python + system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY configs/ configs/
COPY src/ src/

# Dataset gets mounted at runtime
VOLUME /app/data

# Default: run all experiments
ENTRYPOINT ["python", "src/run_experiments.py"]
CMD ["--experiment", "all"]
