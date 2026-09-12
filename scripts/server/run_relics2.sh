#!/usr/bin/env bash
set -euo pipefail

# Resolve every path from this tracked script. No Windows path or login-time
# working directory is part of the analysis configuration.
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
virtual_environment="${STERILE_VENV:-$HOME/data/venvs/sterile-py311}"

if [[ ! -x "$virtual_environment/bin/python" ]]; then
    echo "Python environment not found: $virtual_environment" >&2
    exit 2
fi

# Prevent one Python worker from silently creating a second layer of BLAS
# threads. A caller may override these values explicitly when appropriate.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$repository_root"
exec "$virtual_environment/bin/python" -B run.py "$@"
