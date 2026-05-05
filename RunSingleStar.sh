#!/bin/bash
set -e

# =====================================
# USER SETTINGS
# =====================================

STAR_NAME="$1"        # passed as argument
FSCALE="$2"

SPECTRA_DIR="$HOME/atmos/BA/processed_spectra"
PLANET_TEMPLATE="$HOME/atmos/BA/planets/GOOD_PLANET.dat"

INPUT_DIR="PHOTOCHEM/INPUTFILES"
OUTPUT_BASE="OutputStorage"

# =====================================
# SAFETY CHECKS
# =====================================

if [ -z "$STAR_NAME" ] || [ -z "$FSCALE" ]; then
    echo "Usage:"
    echo "./RunSingleStar.sh <starfile.dat> <fscale>"
    exit 1
fi

STAR_FILE="$SPECTRA_DIR/$STAR_NAME"

if [ ! -f "$STAR_FILE" ]; then
    echo "ERROR: stellar file not found:"
    echo "$STAR_FILE"
    exit 1
fi

RUN_NAME="${STAR_NAME%.dat}_F${FSCALE}"
RUN_DIR="$OUTPUT_BASE/$RUN_NAME"

mkdir -p "$RUN_DIR/INPUT"
mkdir -p "$RUN_DIR/PHOTOCHEM_OUTPUT"
mkdir -p "$RUN_DIR/CLIMA_OUTPUT"

echo "================================"
echo "Running:"
echo "Star   : $STAR_NAME"
echo "FSCALE : $FSCALE"
echo "================================"

# =====================================
# INITIALISE ATMOS TEMPLATE
# =====================================

printf '%s\n' "ArcheanSORG+haze" "n" "n" "n" "n" | ./RunModels.sh

# =====================================
# INJECT STELLAR SPECTRUM
# =====================================

echo "Injecting stellar spectrum..."

cp "$STAR_FILE" "$INPUT_DIR/stellar.dat"

# =====================================
# SET PLANET FILE
# =====================================

cp "$PLANET_TEMPLATE" "$INPUT_DIR/PLANET.dat"

sed -i -E \
"s/^[[:space:]]*[0-9.+-Ee]+[[:space:]]*= FSCALE/${FSCALE} = FSCALE/" \
"$INPUT_DIR/PLANET.dat"

# =====================================
# VERIFY INPUTS (CRITICAL)
# =====================================

echo
echo "INPUT CHECK:"
md5sum "$INPUT_DIR/stellar.dat"
grep FSCALE "$INPUT_DIR/PLANET.dat"
echo

# =====================================
# CLEAN OLD OUTPUTS
# =====================================

rm -f PHOTOCHEM/OUTPUT/*
rm -f CLIMA/IO/*

# =====================================
# RUN ATMOS
# =====================================

script -q -c "./RunAtmos.sh"

# =====================================
# SAVE RESULTS
# =====================================

cp PHOTOCHEM/OUTPUT/* "$RUN_DIR/PHOTOCHEM_OUTPUT/" 2>/dev/null || true
cp CLIMA/IO/* "$RUN_DIR/CLIMA_OUTPUT/" 2>/dev/null || true
cp PHOTOCHEM/INPUTFILES/* "$RUN_DIR/INPUT/"

echo
echo "✅ COMPLETE: $RUN_NAME"
