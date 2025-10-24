#  WSI-Babel-Shark 
> Advanced, modular, and fully Dockerized metadata extraction pipeline for digital pathology slides (SVS, NDPI, PNG).  
> Successor to **WSI-BabelFish**, redesigned and extended at the *Computational Pathology Section, Institute of Pathology, Heidelberg University.*

---

##  Introduction

**WSI-Babel-Shark** is a next-generation digital pathology pipeline designed to **automatically extract, decode, and organize metadata** from whole-slide images (WSIs).  
It provides a fully modular, configuration-driven architecture for laboratories, research groups, and clinical archives that manage large-scale histopathology datasets.

Building on the foundation of **WSI-BabelFish**, Babel-Shark introduces:
- A unified **CLI and YAML-driven orchestration layer** for reproducible workflows  
- Integrated **OCR, DataMatrix, and ROI-fallback engines** for resilient metadata recovery  
- A modular **layout classifier** that automates label segmentation  
- An optional **ROI-Annotator GUI tool** for manual annotation and model training  
- Fully **Dockerized deployment**, ensuring consistent results across environments  
- Native support for **SQLite + CSV registries** for traceable data management  

By combining computer vision, text recognition, and configurable data processing,  
**Babel-Shark** transforms unstructured slide labels into structured, machine-readable metadata — enabling seamless integration into Laboratory Information Systems (LIS), research databases, and AI-driven pipelines.

---


## Key Modules

| Module | Purpose | CLI Command |
|--------|----------|-------------|
| `collect` | Collect slides from `watch_dir` into staging folder | `python -m babel_shark.cli collect` |
| `label` | Crop label/macro regions from WSIs | `python -m babel_shark.cli label` |
| `dm` | Read and decode DataMatrix codes | `python -m babel_shark.cli dm` |
| `extract` | ROI-based metadata extraction | `python -m babel_shark.cli extract` |
| `stain` | Detect stain type (H&E, PAS, Giemsa…) | `python -m babel_shark.cli stain` |
| `slideid` | Generate and rename final slide IDs | `python -m babel_shark.cli slideid` |
| `db` | Update SQLite + CSV registry | `python -m babel_shark.cli db` |
| `pipeline` | Run the full orchestrator | `python -m babel_shark.cli pipeline run` |

---

## Dockerized Execution

Pull prebuilt Docker image:
```bash
docker pull bioinfo2026/wsi-babel-shark:1.0.0
```

Run pipeline directly:
```bash
docker run --rm -v "$PWD":/app -w /app bioinfo2026/wsi-babel-shark:1.0.0   pipeline run --config ./config/config.default.yaml
```

### Docker Compose (optional)
```yaml
services:
  pipeline:
    image: bioinfo2026/wsi-babel-shark:1.0.0
    working_dir: /app
    volumes:
      - ./:/app
    environment:
      - TZ=Europe/Berlin
    command: ["--help"]
```

Run via compose:
```bash
docker compose run --rm pipeline pipeline run --config ./config/config.default.yaml
```

---

## Layout Classifier (Index Builder)

The **Layout Classifier** is a CNN-based submodule for automatic *label region detection and layout classification*.  
It automates the placement of bounding boxes for CaseID, BlockID, and Stain areas — replacing manual ROI definitions.

### Steps

#### 1️⃣ Build dataset index
```bash
python -m babel_shark.layout.layout_cli build --config ./config/layout.yaml
```
Generates train/validation file lists.

#### 2️⃣ Train classifier
```bash
python -m babel_shark.layout.layout_cli learn --config ./config/layout.yaml
```
Trains a lightweight CNN model; output saved to:
```
model/index.ndpz
```

#### 3️⃣ Validate or Predict
```bash
python -m babel_shark.layout.layout_cli validate --config ./config/layout.yaml
python -m babel_shark.layout.layout_cli predict --config ./config/layout.yaml
```

---

##  ROI Annotator App

The **ROI Annotator** is a lightweight, cross-platform GUI tool for manually creating *Regions of Interest (ROIs)*.  
Originally developed as a companion to WSI-Babel-Shark, it’s also suitable for independent annotation tasks.

### Features
- Draw rectangular ROIs on `.png`, `.jpeg`, `.tiff`
- Assign custom labels (e.g., CaseID, PotID, Stain)
- Save and load annotations as `.json`
- Export:
  - **Average Image** (clean)
  - **Overlay Image** (with boxes)
- No external dependencies — pure Python (Tkinter + Pillow + NumPy)

### Quick Start

Download prebuilt binaries from the  
👉 [ROI-Annotator Releases page](https://github.com/Shahram-Bioinfo/ROI-Annotator/releases)

#### Windows
1. Extract ZIP → `ROI-Annotator.exe`
2. Run it directly (ignore SmartScreen warning)

#### macOS
```bash
xattr -dr com.apple.quarantine "/path/to/ROI-Annotator.app"
open ROI-Annotator.app
```

#### Linux
```bash
chmod +x ROI-Annotator
./ROI-Annotator
```

#### Build from source
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pyinstaller pillow numpy
pyinstaller --onefile --windowed ROI.Annotator.py
```

###  Output
- `*.json` — ROI coordinates and labels  
- `*_overlay.png` — annotated visualization for QC  

---

##  Pipeline Output Structure

| Path | Description |
|------|--------------|
| `output/<timestamp>/pipeline_*.log` | Log files |
| `output/slide_metadata.csv` | Master CSV metadata table |
| `output/*.sqlite` | SQLite registry database |
| `output/*_results.xlsx` | Intermediate Excel outputs |
| `data/caseid/<CaseID>/` | Final renamed slides |
| `docs/figures/babel-shark-pipeline-flow.png` | (Pipeline flowchart placeholder) |

---

## Typical Workflow

```bash
# Step 1 – Collect slides
python -m babel_shark.cli collect --config ./config/config.default.yaml

# Step 2 – Crop labels
python -m babel_shark.cli label --config ./config/config.default.yaml

# Step 3 – Decode DataMatrix
python -m babel_shark.cli dm --config ./config/config.default.yaml

# Step 4 – Run stain detection
python -m babel_shark.cli stain --config ./config/config.default.yaml

# Step 5 – Rename & update registry
python -m babel_shark.cli slideid --config ./config/config.default.yaml

# Step 6 – Run full pipeline (automated)
python -m babel_shark.cli pipeline run --config ./config/config.default.yaml
```

---

## Flowchart

📘 **Pipeline Overview**
(placeholder for figure — save your diagram at:  
`docs/figures/babel-shark-pipeline-flow.png`)

```
[Collect] → [Label Extractor] → [DataMatrix Reader]
    ↓                  ↓
 [ROI Annotator] → [Layout Classifier]
    ↓                  ↓
 [Stain Detector] → [SlideID Creator]
    ↓
 [CSV + SQLite Registry]
```

---

## Citation & Acknowledgment

If you use this tool, please cite:

> Aliyari S, Weis C-A. *WSI-Babel-Shark: Modular Dockerized Metadata Extraction Pipeline for Digital Pathology Slides.*  
> Computational Pathology Section, Institute of Pathology, Heidelberg University (2025).

---

## Authors

**Shahram Aliyari**  
Computational Pathology, Institute of Pathology Heidelberg  
📧 aliyari.shahram@uni-heidelberg.de  
📧 aliyari.shahram@gmail.com  

**Supervisor:**  
**Dr. Cleo-Aron Weis** — cleo-aron.weis@uni-heidelberg.de  

---

## 📄 License
Released under the **MIT License**.

---

## 🔗 Related Resources

- [WSI-BabelFish (original version)](https://github.com/ComputationalPathologyHeidelberg/WSI-BabelFish)  
- [ROI-Annotator App](https://github.com/Shahram-Bioinfo/ROI-Annotator)  
- [Docker Hub Image](https://hub.docker.com/r/bioinfo2026/wsi-babel-shark)

---

## Repository Structure
```
WSI-Babel-Shark/
├─ config/                → YAML config files
├─ data/                  → Input slides (test, allslides, caseid)
├─ docs/figures/          → Documentation figures
├─ model/                 → Trained layout models
├─ output/                → Logs, results, registry
├─ roiset/                → ROI annotator outputs
├─ src/babel_shark/       → Core source code
├─ tests/                 → Pytest unit tests
├─ Dockerfile             → Container build file
├─ docker-compose.yml     → Compose orchestration
└─ README.md              → This file
```

---

##  Future Directions
- GPU-accelerated OCR integration (EasyOCR + MistralOCR hybrid)
- RESTful microservice deployment via FastAPI
- Extended stain classifier fine-tuning (ResNet-based)
- Continuous integration & validation datasets

---

*© 2025 Computational Pathology Heidelberg (CPH). All rights reserved.*
