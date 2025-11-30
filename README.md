# WSI-Babel-Shark

> Advanced, modular, and fully Dockerized metadata extraction pipeline for digital pathology slides (SVS, NDPI, PNG).  
> Successor to **WSI-BabelFish**, redesigned and extended at the *Computational Pathology Section, Institute of Pathology, Heidelberg University.*

---

## Introduction

**WSI-Babel-Shark** is a next-generation digital pathology pipeline designed to **automatically extract, decode, and organize metadata** from whole-slide images (WSIs).  
It provides a fully modular, configuration-driven architecture for laboratories, research groups, and clinical archives that manage large-scale histopathology datasets.

Building on the foundation of **WSI-BabelFish**, Babel-Shark introduces:

- A unified **CLI and YAML-driven orchestration layer**  
- Integrated **OCR, DataMatrix, and ROI-fallback engines**  
- A modular **layout classifier**  
- Optional **ROI-Annotator GUI**  
- Fully **Dockerized deployment**  
- Integrated **SQLite + Excel/CSV registry**  

This pipeline transforms unstructured slide labels into structured metadata and is designed for both research and clinical digital pathology workflows.

---

## Installation (Non-Docker / Local)

### 1. Clone the repository

```bash
git clone https://github.com/cpheidelberg/WSI-Babel-Shark.git
cd WSI-Babel-Shark
```

### 2. Create and activate virtual environment

#### Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

#### Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Run pipeline locally

```bash
babel-shark-cli pipeline run --config ./config/config.yaml
```

---

## Key Modules

| Module | Purpose | CLI |
|--------|---------|------|
| `collect` | Collect slides from watch directory | `babel-shark-cli collect` |
| `label` | Extract label region | `babel-shark-cli label` |
| `dm` | Decode DataMatrix | `babel-shark-cli dm` |
| `extract` | ROI metadata extraction | `babel-shark-cli extract` |
| `stain` | Stain classification | `babel-shark-cli stain` |
| `slideid` | Generate SlideID | `babel-shark-cli slideid` |
| `db` | SQLite/Excel registry | `babel-shark-cli db` |
| `pipeline` | Full orchestrator | `babel-shark-cli pipeline run` |

---

## Docker Usage

### Pull the image

```bash
docker pull bioinfo2026/wsi-babel-shark:latest
```

### Run the pipeline

```bash
docker run --rm   -v /path/to/slides:/slides   -v /path/to/output:/output   -v /path/to/config:/config   bioinfo2026/wsi-babel-shark:latest     pipeline run --config /config/config.docker.yaml
```

### Run individual modules

```bash
docker run --rm -v "/slides" -v "/output" -v "/config"   bioinfo2026/wsi-babel-shark:latest collect run --config /config/config.docker.yaml
```

---

## Layout Classifier

The layout classifier automates ROI placement by identifying label structures.

### Build index

```bash
python -m babel_shark.layout.layout_cli build --config ./config/layout.yaml
```

### Train

```bash
python -m babel_shark.layout.layout_cli learn --config ./config/layout.yaml
```

### Predict

```bash
python -m babel_shark.layout.layout_cli predict --config ./config/layout.yaml
```

---

## ROI Annotator App

Lightweight GUI for manual ROI creation and dataset curation.

Download:  
https://github.com/Shahram-Bioinfo/ROI-Annotator/releases

---

## Pipeline Output Structure

- `output/<timestamp>/pipeline_*.log`  
- `output/slide_metadata.xlsx`  
- `output/WSI-Babel-Shark.sqlite`  
- `data/caseid/<CaseID>/`  
- `output/roi_debug/`  

---

## Typical Workflow (Local)

```bash
babel-shark-cli collect run --config ./config/config.yaml
babel-shark-cli label run --config ./config/config.yaml
babel-shark-cli dm run --config ./config/config.yaml
babel-shark-cli extract run --config ./config/config.yaml
babel-shark-cli stain run --config ./config/config.yaml
babel-shark-cli slideid run --config ./config/config.yaml
babel-shark-cli pipeline run --config ./config/config.yaml
```

---

## Flowchart Diagram

Place your flowchart in:

```
docs/figures/babel-shark-pipeline-flow.png
```

Then reference it in README:

```markdown
![WSI-Babel-Shark pipeline overview](docs/figures/babel-shark-pipeline-flow.png)
```

---

## Citation

Aliyari S, Wagner M, Weis C-A. **WSI-Babel-Shark: Modular Dockerized Metadata Extraction Pipeline for Digital Pathology Slides.**  
Computational Pathology Section, Institute of Pathology, Heidelberg University (2025).

---

## Authors

**Shahram Aliyari** — First Author  
**Maximilian Wagner** — Co-Author  
**Dr. Cleo-Aron Weis** — Supervisor  

---

## License

MIT License.

---

## Related Tools

- ROI-Annotator: https://github.com/Shahram-Bioinfo/ROI-Annotator  
- WSI-BabelFish: https://github.com/ComputationalPathologyHeidelberg/WSI-BabelFish  
- Docker Hub: https://hub.docker.com/r/bioinfo2026/wsi-babel-shark  
---

*© 2025 Computational Pathology Heidelberg (CPH). All rights reserved.*
