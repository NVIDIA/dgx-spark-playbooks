---
name: dgx-spark-single-cell
description: An end-to-end GPU-powered workflow for scRNA-seq using RAPIDS — on NVIDIA DGX Spark. Use when setting up single-cell on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/single-cell/README.md -->
# Single-cell RNA Sequencing

> An end-to-end GPU-powered workflow for scRNA-seq using RAPIDS

Single-cell RNA sequencing (scRNA-seq) lets researchers study gene activity in each cell on its own, exposing variation, cell types, and cell states that bulk methods hide. But these large, high-dimensional datasets take heavy compute to handle.

This playbook shows an end-to-end GPU-powered workflow for scRNA-seq using [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a RAPIDS powered library in the [scverse® ecosystem](https://github.com/scverse). It follows the familiar [Scanpy API](https://scanpy.readthedocs.io/en/stable/) and lets researchers run the steps of data preprocessing, quality control (QC) and cleanup, visualization, and investigation faster than CPU tools by working with sparse count matrices directly on the GPU.

**Outcome**: 1. GPU-Accelerated Data Loading & Preprocessing
2. QC cells visually to understand the data
3. Filter unusual cells
4. Remove unwanted sources of variation 
5. Cluster and visualize PCA and UMAP data
6. Batch Correction and analysis using Harmony, k-nearest neighbors, UMAP, and tSNE
7. Explore the biological information from the data with differential expression analysis and trajectory analysis

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/single-cell/README.md`
<!-- GENERATED:END -->
