# Step 1: Lightweight Python base image
FROM python:3.12-slim

# Step 2: System dependencies for OpenSlide, OpenCV, Tesseract, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        openslide-tools \
        libopenslide0 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Step 4: Create non-root user
RUN useradd -ms /bin/bash babelshark

# Step 5: Application working directory
WORKDIR /app

# Step 6: Copy project metadata, source, and assets
COPY pyproject.toml /app/
COPY src /app/src
COPY config /app/config
COPY model /app/model
COPY roiset /app/roiset

# Step 7: Install Python dependencies

# 7.a) Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 7.b) Install PyTorch + torchvision (CPU wheels)
RUN pip install --no-cache-dir \
      torch \
      torchvision \
      --index-url https://download.pytorch.org/whl/cpu

# 7.c) Install other third-party dependencies
RUN pip install --no-cache-dir \
      numpy \
      pandas \
      pyyaml \
      pillow \
      openslide-python \
      opencv-python-headless \
      pylibdmtx \
      easyocr \
      reportlab \
      tqdm \
      rapidfuzz \
      openpyxl \
      XlsxWriter \
      scikit-learn \
      matplotlib \
      seaborn

# 7.d) Install the WSI-Babel-Shark package itself
RUN pip install --no-cache-dir /app

# Step 8: Create runtime directories for data and config
RUN mkdir -p /data/input /data/output /config && \
    chown -R babelshark:babelshark /app /data /config

# Step 9: Switch to non-root user
USER babelshark

# Step 10: Runtime working directory (host volumes will be mounted here)
WORKDIR /data

# Step 11: Default config path (can be overridden by CLI or ENV)
ENV BABEL_SHARK_CONFIG=/config/config.docker.yaml

# Step 12: Entrypoint – unified CLI
ENTRYPOINT ["python", "-m", "babel_shark.cli"]
CMD ["--help"]
