#!/usr/bin/env bash
# Rebuild every per-tract feature table from the public challenge bucket, then write the
# submission file(s). Needs network access (the data is read in place from Source Cooperative).
#
#   ./run_all.sh            # features + the default submission
#   ./run_all.sh v2b v3d    # features + the listed variants (see VARIANTS in src/model.py)
#   SKIP_FEATURES=1 ./run_all.sh v3d   # reuse features/ already built
#
# Set PYTHON to the interpreter of the environment built from requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python}
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(v2b)

step() { local t0=$SECONDS; "$PY" "$@"; echo "[run_all] $1 done in $((SECONDS - t0)) s"; }

if [ -z "${SKIP_FEATURES:-}" ]; then
  step features.py     # buildings by source, places by category, ACS, strata per tract
  step roads.py        # road length by class x route network, inside and along each boundary
  step zips.py         # places by postcode, buildings by footprint size
  step evidence.py     # OSM fire-station / school buildings, fire stations filed under other categories
fi
for v in "${VARIANTS[@]}"; do
  step model.py "$v"
done
