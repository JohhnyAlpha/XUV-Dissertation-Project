# Project README

## Overview

This repository contains scripts and output plots associated with modelling atmospheric evolution and habitability of Earth-like exoplanets around K-dwarf stars. The work focuses on stellar XUV evolution, atmospheric escape, photochemistry, and climate interactions using a coupled modelling approach.

---

## Repository Structure

```
.
├── scripts/        # All code used for modelling and analysis
└── README.md       # Project documentation (this file)
└── *.sh            # Shell scripts (not used) for automating ATMoS```

---

## Scripts

The `scripts/` directory contains the code used throughout the project. This may include:

* Data processing scripts
* Stellar spectrum generation
* Model execution workflows (e.g., Atmos runs)
* Plotting and analysis tools

Each script should ideally be documented with comments explaining its purpose and usage
However due to time constraints comments are limited.

---

## Plots

All generated figures are stored in:

```
plots.zip
```

This archive contains:

* Output plots from simulations
* Figures used in reports or dissertation work

To view the plots, extract the archive:

```bash
unzip plots.zip
```

---

## Usage

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   ```

2. Navigate into the project:

   ```bash
   cd <repo-name>
   ```

3. Run scripts from the `scripts/` directory as required.

---

## Notes

* Some scripts may depend on external tools (e.g., MATLAB, Atmos, or Python libraries).
* Ensure required dependencies are installed before running.
* Large output files (such as plots) are stored as compressed archives to keep the repository size manageable.

---

## Future Improvements

* Expand documentation for individual scripts
* Separate raw data, processed data, and outputs into dedicated directories
* Replace zipped plots with structured folders if repository size allows

---

## Author

Bernard Atkinson

---
