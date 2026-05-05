#!/bin/bash

############################################
# ATMoS STAR RUN SCRIPT
# Follows RunAtmos.sh logic exactly
# Only injects stellar spectrum
############################################

ATMOS_DIR="$HOME/atmos"
SPECTRA_DIR="$HOME/atmos/BA/processed_spectra"

############################################
# SELECT STAR SPECTRUM (ONLY LINE TO EDIT)
############################################

STARFILE="EpsilonEri_quiet.dat"

############################################

echo "===================================="
echo "Running ATMoS with:"
echo "$STARFILE"
echo "===================================="

# ----------------------------------
# Inject stellar spectrum
# ----------------------------------

echo "Injecting stellar spectrum..."

cp "$SPECTRA_DIR/$STARFILE" \
"$ATMOS_DIR/PHOTOCHEM/DATA/spectrum.dat"

cp "$SPECTRA_DIR/$STARFILE" \
"$ATMOS_DIR/CLIMA/DATA/spectrum.dat"

echo "Spectrum injected."

# ----------------------------------
# Run original ATMoS workflow
# ----------------------------------

cd $ATMOS_DIR

./RunAtmos.sh

echo ""
echo "===================================="
echo "RUN COMPLETE"
echo "===================================="
