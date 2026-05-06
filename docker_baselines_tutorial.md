# Running PAN'25 Generative AI Authorship Verification Baselines via Docker

This tutorial will guide you through running the baselines for the PAN 2025 "Voight-Kampff" Generative AI Authorship Verification task using Docker. Using Docker is highly recommended as it comes pre-configured with all necessary dependencies, including GPU support for models that require it.

## Prerequisites

Before you start, make sure you have:
1. **Docker Installed**: Ensure Docker is installed and running on your system.
2. **NVIDIA Container Toolkit (Optional but recommended)**: If you plan to run deep learning baselines (like `binoculars`), you need an NVIDIA GPU and the NVIDIA Container Toolkit to pass your GPU to the container (`--gpus=all`).
3. **Your Data File**: Your dataset should be in `.jsonl` format. In this tutorial, we will refer to it as `val.jsonl`.

## Available Baselines

The provided Docker image supports the following baselines:
- `tfidf`: TF-IDF Support Vector Machine (SVM).
- `ppmd`: Compression-based cosine similarity.
- `binoculars`: Binoculars (Requires GPU).

## General Command Structure

To evaluate a baseline on your data, use the following Docker command structure:

```bash
docker run --rm --gpus=all \
    -v /absolute/path/to/your/input.jsonl:/val.jsonl \
    -v /absolute/path/to/output_directory:/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    <BASELINENAME> /val.jsonl /out
```

### Explanation of flags:
- `--rm`: Automatically removes the container after the process finishes.
- `--gpus=all`: Exposes all available GPUs to the container. (Only strictly necessary for `binoculars`).
- `-v /absolute/path/to/your/input.jsonl:/val.jsonl`: Mounts your local input `.jsonl` file into the container at `/val.jsonl`. Make sure to provide an absolute path for your local file.
- `-v /absolute/path/to/output_directory:/out`: Mounts your local output directory to the container at `/out`. The baseline predictions will be saved here.
- `<BASELINENAME>`: Replace this with the baseline you wish to run (`tfidf`, `ppmd`, or `binoculars`).

## Examples

### 1. Running the TF-IDF Baseline

This baseline is lightweight and does not strictly require a GPU. Based on your folder structure, your data is in the `data/` directory. Assuming you want outputs in the current directory:

**On Windows (PowerShell):**
```powershell
docker run --rm -v ${PWD}\data\val.jsonl:/val.jsonl -v ${PWD}:/out `
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
    tfidf /val.jsonl /out
```

**On Linux / macOS:**
```bash
docker run --rm -v $(pwd)/data/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    tfidf /val.jsonl /out
```

### 2. Running the PPMd Baseline

This is a compression-based cosine similarity baseline. It is also lightweight and does not require a GPU.

**On Windows (PowerShell):**
```powershell
docker run --rm -v ${PWD}\data\val.jsonl:/val.jsonl -v ${PWD}:/out `
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
    ppmd /val.jsonl /out
```

**On Linux / macOS:**
```bash
docker run --rm -v $(pwd)/data/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    ppmd /val.jsonl /out
```

### 3. Running the Binoculars Baseline

The Binoculars baseline requires a GPU for reasonable performance. 

**On Windows (PowerShell):**
```powershell
docker run --rm --gpus=all -e HF_HOME=/hf_cache -v ${PWD}\.hf_cache:/hf_cache -v ${PWD}\data\val.jsonl:/val.jsonl -v ${PWD}:/out `
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
    binoculars /val.jsonl /out
```

**On Linux / macOS:**
```bash
docker run --rm --gpus=all -e HF_HOME=/hf_cache -v $(pwd)/.hf_cache:/hf_cache -v $(pwd)/data/val.jsonl:/val.jsonl -v $(pwd):/out \
    ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
    binoculars /val.jsonl /out
```

## Retrieving Results

After the Docker container finishes execution, it will automatically shut down and remove itself (thanks to the `--rm` flag). You can find the output predictions inside the local directory that you mounted to `/out` (e.g., your current working directory in the examples above).
