#!/bin/bash

############################################
# SAFE SINGLE ATMoS RUN SCRIPT
# Dissertation-safe version
############################################

set -e

ATMOS_DIR="$HOME/atmos"
SPECTRA_DIR="$HOME/atmos/BA/processed_spectra"
OUTPUT_DIR="$HOME/atmos/OUTPUTS"

echo "================================="
echo "ATMoS Controlled Run"
echo "================================="

# -----------------------------
# Choose spectrum
# -----------------------------
echo ""
echo "Available spectra:"
ls $SPECTRA_DIR/*.dat
echo ""

read -p "Enter spectrum filename: " STARFILE
read -p "Enter run name: " RUNNAME

FULLSPEC="$SPECTRA_DIR/$STARFILE"

if [ ! -f "$FULLSPEC" ]; then
    echo "ERROR: Spectrum not found"
    exit 1
fi

echo ""
echo "Using spectrum:"
echo "$FULLSPEC"

# -----------------------------
# Inject stellar spectrum
# -----------------------------
echo "Injecting spectrum..."

cp "$FULLSPEC" \
"$ATMOS_DIR/PHOTOCHEM/DATA/spectrum.dat"

echo "Spectrum installed."

# -----------------------------
# Compile Photochem
# -----------------------------
echo ""
echo "Compiling Photochem..."

cd $ATMOS_DIR/PHOTOCHEM
make clean
make

# -----------------------------
# Run Photochem
# -----------------------------
echo ""
echo "Running Photochem..."

./Photochem

echo "Photochem complete."

# -----------------------------
# Compile Clima
# -----------------------------
echo ""
echo "Compiling Clima..."

cd $ATMOS_DIR/CLIMA
make clean
make

# -----------------------------
# Run Clima
# -----------------------------
echo ""
echo "Running Clima..."

./Clima

echo "Clima complete."

# -----------------------------
# Archive outputs
# -----------------------------
echo ""
echo "Saving outputs..."

RUN_OUT="$OUTPUT_DIR/$RUNNAME"
mkdir -p "$RUN_OUT"

cp $ATMOS_DIR/*.out "$RUN_OUT" 2>/dev/null || true
cp $ATMOS_DIR/*.tab "$RUN_OUT" 2>/dev/null || true
cp $ATMOS_DIR/PHOTOCHEM/*.out "$RUN_OUT" 2>/dev/null || true
cp $ATMOS_DIR/CLIMA/*.out "$RUN_OUT" 2>/dev/null || true

echo "Outputs saved to:"
echo "$RUN_OUT"

echo ""
echo "================================="
echo "RUN COMPLETE SUCCESSFULLY"
echo "================================="
