# =====================================================
# Dockerfile — WSI-Babel-Shark Pipeline (CPU)
# Base: slim Python + system libs for OpenCV, OpenSlide, DataMatrix, PDF
# =====================================================

# --- 1) Base image ---
FROM python:3.11-slim

# --- 2) System dependencies ---
# Required for:
# - OpenCV (image I/O)
# - OpenSlide (.svs / .ndpi reading)
# - DataMatrix decoding (libdmtx0t64)
# - PDF generation (ReportLab)
# - Image codecs (JPEG, TIFF, JP2)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libjpeg62-turbo \
    libopenjp2-7 \
    libtiff6 \
    libopenslide0 \
    openslide-tools \
    libdmtx0t64 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- 3) Python runtime hygiene ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- 4) Working directory ---
WORKDIR /app

# --- 5) Install Python dependencies first (for better caching) ---
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# --- 6) Copy project source code ---
COPY . /app

# --- 7) Make src importable ---
ENV PYTHONPATH=/app/src

# --- 8) Default entrypoint for the pipeline ---
ENTRYPOINT ["python", "-m", "babel_shark.cli"]

# --- 9) Default command (runs when no args given) ---
CMD ["--help"]
