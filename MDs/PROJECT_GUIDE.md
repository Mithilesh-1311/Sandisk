# ML for IR Drop: Project Guide

## 1. Overview

This repository hosts the benchmark generation and dataset tooling used in the ICCAD 2023 contest problem on static IR drop prediction using machine learning.

The project is focused on generating realistic current maps and power-delivery scenarios, then solving or analyzing voltage drop behavior using a custom IR-drop simulation pipeline. It includes:

- synthetic current map generation,
- technology-specific benchmark generation,
- reference/realistic design maps,
- IR solver and grid infrastructure,
- local visualization tools for inspecting generated outputs.

The original README is intentionally short; this file expands the operational details and local workflow needed to run and inspect the project.

---

## 2. Goal of the repository

The goal is to support modeling of static IR drop in power distribution networks (PDNs), especially in the context of machine learning-based prediction. In practice, the repo provides inputs and benchmark datasets used to train/evaluate prediction models by generating or exposing current maps and their associated voltage-drop characteristics.

The data is structured around current maps and IR-drop-related metrics such as:

- current maps,
- effective distance maps,
- IR drop maps,
- PDN density maps,
- voltage maps,
- solver-generated outputs.

---

## 3. Repository structure

```text
ML-for-IR-drop/
├── README.md
├── PROJECT_GUIDE.md
├── LICENSE
├── doc/
│   ├── contest-description.pdf
│   ├── invited-paper.pdf
│   └── ICCAD23-Contest-ProblemC.pdf
├── benchmarks/
│   ├── fake-circuit-data/
│   ├── hidden-real-circuit-data/
│   └── real-circuit-data/
├── src/
│   ├── current_mapgen.py
│   ├── generate_benchmark_maps.py
│   ├── generate_benchmark_maps_asap7.py
│   ├── generate_benchmark_maps_nangate45.py
│   ├── generate_gan_maps_nangate45.py
│   ├── grid.py
│   ├── ir_solver.py
│   └── node.py
├── viewer/
│   ├── app.js
│   ├── build_manifest.py
│   ├── datasets.json
│   └── index.html
└── ...
```

### Main directories

- `benchmarks/` contains benchmark samples and current/IR map data.
- `src/` contains the core generation and solving logic.
- `viewer/` contains the lightweight local browser-based heatmap viewer.
- `doc/` contains contest materials and papers.

---

## 4. Key scripts and what they do

### 4.1 `src/current_mapgen.py`

This is the smallest runnable script and acts as a proof-of-life generator for synthetic current maps.

It does the following:

- creates a random Gaussian field,
- normalizes it to a current range,
- converts the map to a DOK-style sparse representation,
- prints timing information.

This is the script I verified successfully with Python:

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Gitu\ML-for-IR-drop"
& "C:/Program Files/Python313/python.exe" src/current_mapgen.py
```

It completed successfully and printed:

```text
start_time 0.000002
create_maps time 1.991817
convert to dok 0.078749
```

This confirms the core environment and dependencies are working.

---

### 4.2 `src/generate_benchmark_maps.py`

This is a broader benchmark-generation script. It orchestrates:

- current map loading,
- design processing,
- voltage generation,
- region generation,
- IR solve steps,
- output generation to CSV and SPICE-style netlists.

It uses functions such as:

- `load_current_maps()`
- `load_params()`
- `generate_outputs()`
- `generate_vsrc()`
- `generate_regions()`
- `solve_ir()`
- `generate_spice()`

This script is the backbone for producing benchmark-style outputs for downstream analysis.

---

### 4.3 `src/generate_benchmark_maps_asap7.py`

This is a technology-specific variant for the ASAP7 technology node. It is likely used to generate design/current benchmark files under a specific process corner and technology constraints.

---

### 4.4 `src/generate_benchmark_maps_nangate45.py`

This is the Nangate45 variant of the benchmark generation pipeline.

---

### 4.5 `src/generate_gan_maps_nangate45.py`

This script is associated with generated map creation from GAN-generated image content, likely used to augment current maps in an ML pipeline.

---

### 4.6 `src/ir_solver.py`

This is the numerical core of the project. It handles PDN solving for current distribution and voltage drop.

It contains logic for:

- representing nodes and grid connectivity,
- updating current and voltage states,
- solving IR drop over regions,
- producing outputs for plots or analysis.

This is the file that links current map generation with electrical behavior.

---

### 4.7 `src/grid.py` and `src/node.py`

These define the underlying grid and node structures used in the IR solver.

In effect, they represent the power grid as a network of connected electrical nodes with properties such as:

- positions,
- layers,
- resistive links,
- boundary and region conditions.

---

## 5. Data format and benchmark content

The `benchmarks/fake-circuit-data/` directory contains files like:

```text
current_map00_current.csv
current_map00_eff_dist.csv
current_map00_ir_drop.csv
current_map00_pdn_density.csv
```

These correspond to different representations of the same physical layout or generated design scenario.

The values are numeric matrices saved as CSVs, which makes them easy to inspect in:

- Excel,
- Python/NumPy,
- local browser viewers,
- custom plotting scripts.

The sample matrix we verified is dimensions:

```text
shape (821, 821)
min 3.33148e-08
max 3.73379e-07
mean 4.6650027025655704e-08
```

This confirms the bench data is valid and large enough to visualize as heatmaps.

---

## 6. Local environment setup

### 6.1 Python version

The repository is Python-based. A Python 3.10+ environment is suitable. The project was validated under Python 3.13 in this environment.

### 6.2 Required packages

Install the runtime dependencies:

```powershell
& "C:/Program Files/Python313/python.exe" -m pip install gstools tqdm numpy matplotlib scipy scikit-image opencv-python
```

These are the packages seen in the project imports:

- `gstools`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-image`
- `opencv-python`
- `tqdm`

---

## 7. Quick start

### Run the current map generator

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Gitu\ML-for-IR-drop"
& "C:/Program Files/Python313/python.exe" src/current_mapgen.py
```

This creates synthetic random current maps and converts them into DOK representation.

### Start the local visualization server

From the repo root:

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Gitu\ML-for-IR-drop"
& "C:/Program Files/Python313/python.exe" -m http.server 8002 --directory .
```

Then open:

```text
http://localhost:8002/viewer/index.html
```

This loads the viewer and renders benchmark CSV files as heatmaps.

---

## 8. Local browser viewer

The repository includes a lightweight local viewer under `viewer/`.

### Files

- `viewer/index.html` – page structure
- `viewer/app.js` – CSV loader and heatmap renderer
- `viewer/build_manifest.py` – generates `datasets.json`

### How it works

- `build_manifest.py` scans the benchmark folder for CSV files and writes `datasets.json`.
- `index.html` renders a dropdown of datasets.
- `app.js` loads a selected CSV file and paints it to a canvas as a heatmap.
- statistics such as min, max, and mean are displayed alongside the image.

### Important note

The viewer must be served from the repo root, not from the `viewer/` directory alone. This is because the CSV URLs are resolved as:

```text
/benchmarks/fake-circuit-data/current_map00_current.csv
```

If you serve only the `viewer/` folder, the browser cannot resolve those CSVs and the page will appear blank or fail to load data.

---

## 9. Known practical issues and fixes

### 9.1 Missing dependency error

If you see:

```text
ModuleNotFoundError: No module named 'gstools'
```

install the dependencies:

```powershell
& "C:/Program Files/Python313/python.exe" -m pip install gstools tqdm numpy matplotlib scipy scikit-image opencv-python
```

---

### 9.2 Blank screen in browser

This usually happens when:

- the server is started in the wrong working directory,
- the page is served from the `viewer/` folder instead of the repo root,
- the browser is still holding stale page state.

Use:

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Gitu\ML-for-IR-drop"
& "C:/Program Files/Python313/python.exe" -m http.server 8002 --directory .
```

Then open:

```text
http://localhost:8002/viewer/index.html
```

Hard refresh with Ctrl+F5 if needed.

---

## 10. How the project fits into IR drop research

This repository is designed around a common workflow in PDN research:

1. create current distribution maps,
2. map them to an equivalent grid-based network,
3. solve voltage drop under a supply model,
4. produce benchmark-style outputs for machine learning pipelines.

This is especially relevant for the task of predicting static voltage drop without running expensive SPICE-level analysis every time.

---

## 11. What the CSVs represent

Examples from the benchmark data include:

- `*_current.csv` — current density distribution
- `*_eff_dist.csv` — effective distance map
- `*_ir_drop.csv` — solved IR-drop values
- `*_pdn_density.csv` — PDN density information

These are likely used for model training, evaluation, and result comparison in an ML pipeline.

---

## 12. Recommended workflow

For local exploration, a practical workflow is:

1. Create a Python environment.
2. Install dependencies.
3. Run the generator script.
4. Inspect the generated benchmark data.
5. Serve the repo root locally.
6. Open the heatmap viewer.
7. Use the CSV outputs for custom analysis or model development.

This provides a fast loop for:

- validating the data,
- understanding map structure,
- checking solver behavior,
- preparing ML datasets.

---

## 13. Summary

This repository is not a web application by itself, but it is a complete benchmark and solver toolkit for IR-drop prediction work.

It includes:

- synthetic data generation,
- design benchmark generation,
- solver-based voltage drop estimation,
- benchmark output files,
- a local browser viewer for inspecting results.

The simplest successful local path is:

```powershell
cd "C:\Users\Admin\OneDrive\Desktop\Gitu\ML-for-IR-drop"
& "C:/Program Files/Python313/python.exe" -m pip install gstools tqdm numpy matplotlib scipy scikit-image opencv-python
& "C:/Program Files/Python313/python.exe" src/current_mapgen.py
& "C:/Program Files/Python313/python.exe" -m http.server 8002 --directory .
```

Then open:

```text
http://localhost:8002/viewer/index.html
```

This is a complete local workflow for generating and inspecting IR-drop benchmark outputs.
