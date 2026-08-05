# JGR_LunarRegolithPropertiesVariability

Supporting code for the manuscript:

**"Variability of Lunar Regolith Properties and Implications for Interpreting Brightness Temperature Variations"**  

**Corresponding Author**: Carlos Gómez de Olea Ballester  - carlos.olea@tum.de

Submitted to *Journal of Geophysical Research: Planets* (submitted: TBD)

---

## Overview

This repository contains the numerical models, data processing scripts, and analysis routines used in the study of the variability of lunar regolith thermophysical and radiative properties and their implications for interpreting microwave brightness temperature observations.

The repository includes:

- A one-dimensional thermal model used to estimate realistic temperature profiles for lunar regolith.
- Models for thermal, structural and radiative properties of dry and icy lunar regolith.
- A radiative transfer model for calculating microwave brightness temperatures.
- Statistical correlation and sensitivity analyses used to quantify the influence of regolith properties on brightness temperature observations.

The scripts reproduce the analysis and figures presented in the manuscript (Figures 1–13).

---


## Description of files

### `Regolith1D_JGR.py`

Contains the functions required to describe the thermophysical properties of lunar regolith and to run the one-dimensional thermal model.

---

### `RadiativeTransfer.py`

Contains the radiative property parameterisations and microwave radiative transfer model used in the manuscript.
---

### `thermal_model.ipynb`

Jupyter notebook used to run the thermal model simulations.

The notebook performs the iterations required to generate the temperature profiles used in the manuscript and reproduces **Figure 1**.

---

### `generate_figures.ipynb`

Jupyter notebook containing the statistical analysis. Generates **Figures 2–13** presented in the manuscript.

---

## Requirements

The code was developed using Python 3.11

Required Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `seaborn`
- `jupyter`
- `os`
