# Deep Learning Airfoil Optimization Tool

A complete methodology and toolset for the backward design of airfoils based on an optimization algorithm fed by a neural network. This tool utilizes the Class-Shape Transformation (CST) parameterization method and is designed for predicting aerodynamic coefficients ($C_L$, $C_D$, $C_M$) in the 3M to 9M Reynolds number range without the need for manual 3D CFD generation.

## Dataset Acknowledgement
Due to file size limits on GitHub, the heavy `.keras` models, `.pkl` serialized dictionaries, and massive `.csv` dataset files are not tracked in this repository. 

All core data utilized to train the surrogate model and build the baseline catalogs originates from the public dataset: **"Airfoil Computational Fluid Dynamics - 2k shapes, 25 AoA's, 3 Re numbers"**. This high-fidelity database was published by Dakota Ramos et al. (2023) through the Open Energy Data Initiative (OEDI) and the National Renewable Energy Laboratory (NREL) (DOI: 10.25984/2222586). It provides the foundational aerodynamic responses from over 250,000 independent CFD simulations across 1,830 unique airfoil shapes.

## Features
- **Neural Network Surrogate Model:** A Keras Functional API multi-branch Multi-Layer Perceptron (MLP).
- **Optimization Algorithm:** Uses `scipy.optimize.minimize` (SLSQP) to reverse-engineer geometries to meet specific aerodynamic targets.
- **Interactive Interface:** A Flask-based web application to input flight regimes, set aerodynamic targets, and download HTML reports with CAD coordinates.

## Project Structure
- `data/` - Holds datasets and CSV logs (Models are kept locally due to size).
- `scripts/` - Contains the Python orchestrators, MLP training scripts, the optimization engine, and the interface `app.py`.
- `optimization/` - The output directory for the final HTML optimization reports.
