#!/bin/bash
# ==========================================================
# RunAtmosGrid.sh
#
# ATMoS Research-Grade Grid Runner
# Safe parallel execution with isolated environments
#
# Designed for PhotoChem + Clima automation
# ==========================================================

set -euo pipefail

echo "=========================================="
echo " ATMoS GRID RUNNER (Isolated Execution)"
echo "=========================================="
echo

# ----------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------

# CLEAN master atmos install (NEVER run directly)
TEMPLATE="$HOME/atmos/TEMPLATE_ATMOS"

# Where runs will be created
RUNROOT="$HOME/atmos/RUNS"

# Stellar spectra directory
STELLAR_DIR="$HOME/atmos/BA/stellar_scenarios"

# Parallel jobs (set <= CPU cores)
max_jobs=4

# Flux scaling values
fscale_array=(0.75 1.00 1.50)

# ----------------------------------------------------------
# USER INPUT
# ----------------------------------------------------------

read -p "Iterations per case: " iters
echo

mkdir -p "$RUNROOT"

# ----------------------------------------------------------
# BUILD STELLAR CASE LIST
# ----------------------------------------------------------

cases=()
for f in "$STELLAR_DIR"/*.dat; do
    cases+=("$(basename "$f" .dat)")
done

echo "Detected stellar spectra:"
printf '  - %s\n' "${cases[@]}"
echo

# ==========================================================
# FUNCTION: RUN SINGLE CASE
# ==========================================================

run_case () {

    case_name=$1
    fscale=$2

    RUN_DIR="${RUNROOT}/${case_name}_F${fscale}"

    echo
    echo "------------------------------------------"
    echo "Creating run: $case_name  FSCALE=$fscale"
    echo "Directory: $RUN_DIR"
    echo "------------------------------------------"

    # ------------------------------------------------------
    # CREATE ISOLATED COPY OF TEMPLATE
    # ------------------------------------------------------

    cp -r "$TEMPLATE" "$RUN_DIR"
    cd "$RUN_DIR"

    # ------------------------------------------------------
    # VERIFY REQUIRED DIRECTORIES
    # ------------------------------------------------------

    if [[ ! -d PHOTOCHEM/INPUTFILES ]]; then
        echo "ERROR: TEMPLATE_ATMOS structure invalid."
        exit 1
    fi

    # ------------------------------------------------------
    # INJECT STELLAR SPECTRUM
    # ------------------------------------------------------

    cp "${STELLAR_DIR}/${case_name}.dat" \
       PHOTOCHEM/INPUTFILES/stellar.dat

    echo "Injected stellar spectrum:"
    md5sum PHOTOCHEM/INPUTFILES/stellar.dat

    # ------------------------------------------------------
    # MODIFY FSCALE SAFELY
    # ------------------------------------------------------

    sed -i -E \
    "s/^[[:space:]]*[0-9.+-Ee]+[[:space:]]*= FSCALE/${fscale} = FSCALE/" \
    PHOTOCHEM/INPUTFILES/PLANET.dat

    echo "FSCALE confirmation:"
    grep FSCALE PHOTOCHEM/INPUTFILES/PLANET.dat

    # ------------------------------------------------------
    # SAFE CLEANING (DO NOT REMOVE STRUCTURE)
    # ------------------------------------------------------

    echo "Clearing restart state..."

    rm -f CLIMA/IO/*last* 2>/dev/null || true
    rm -f PHOTOCHEM/OUTPUT/* 2>/dev/null || true

    # ------------------------------------------------------
    # ITERATIVE RUN LOOP
    # ------------------------------------------------------

    last_hash=""
    identical_count=0

    for ((iter=1; iter<=iters; iter++)); do

        echo "[$case_name F$fscale] Iteration $iter"

        ./Photo.run > /dev/null
        ./Clima.run > /dev/null

        OUTFILE="PHOTOCHEM/OUTPUT/out.out"

        if [[ ! -s "$OUTFILE" ]]; then
            echo "ERROR: Missing output file."
            exit 1
        fi

        # ---------- HASH CHECK ----------
        current_hash=$(md5sum "$OUTFILE" | awk '{print $1}')

        if [[ "$current_hash" == "$last_hash" ]]; then
            identical_count=$((identical_count+1))
        else
            identical_count=0
        fi

        if [[ $identical_count -ge 3 ]]; then
            echo "Convergence detected — stopping early."
            break
        fi

        last_hash="$current_hash"

        # ---------- SAVE ITERATION SNAPSHOT ----------
        cp "$OUTFILE" \
           "PHOTOCHEM/OUTPUT/out_iter${iter}.out"

        cp CLIMA/IO/clima_allout.tab \
           "CLIMA/IO/clima_iter${iter}.tab" \
           2>/dev/null || true

    done

    # ------------------------------------------------------
    # METADATA LOG
    # ------------------------------------------------------

    cat <<EOF > run_info.txt
Star: $case_name
FSCALE: $fscale
Iterations completed: $iter
Date: $(date)
Run directory: $RUN_DIR
Stellar checksum:
$(md5sum PHOTOCHEM/INPUTFILES/stellar.dat)
EOF

    echo "Run complete: $case_name FSCALE=$fscale"
}

# ==========================================================
# PARALLEL EXECUTION CONTROLLER
# ==========================================================

job_count=0

for case_name in "${cases[@]}"; do
for fscale in "${fscale_array[@]}"; do

    run_case "$case_name" "$fscale" &

    ((job_count++))

    if (( job_count % max_jobs == 0 )); then
        wait
    fi

done
done

wait

echo
echo "=========================================="
echo " ALL RUNS COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Results located in:"
echo "$RUNROOT"
echo
