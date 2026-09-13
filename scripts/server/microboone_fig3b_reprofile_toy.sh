#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
virtual_environment="${STERILE_VENV:-$HOME/data/venvs/sterile-py311}"
output_root="${STERILE_SERVER_OUTPUT_ROOT:-$HOME/data/outputs/microboone_reprofile_toy}"

usage() {
    echo "Usage: $0 start BATCH [WORKERS=28] [TOYS=5000] [FIGURES=both]"
    echo "       $0 status BATCH"
    echo "       $0 watch BATCH [SECONDS=10]"
    echo "       $0 stop BATCH"
}

format_seconds() {
    local seconds=${1%.*}; (( seconds < 0 )) && seconds=0
    printf '%02dd %02dh %02dm %02ds' $((seconds/86400)) $(((seconds%86400)/3600)) $(((seconds%3600)/60)) $((seconds%60))
}

batch_status() {
    local batch=$1 batch_directory="$output_root/$1" manifest="$output_root/$1/manifest.env"
    [[ -f "$manifest" ]] || { echo "Batch not found: $batch_directory" >&2; return 2; }
    source "$manifest"
    local completed=0 file lines
    while IFS= read -r -d '' file; do
        lines=$(wc -l < "$file"); (( lines > 0 )) && completed=$((completed+lines-1))
    done < <(find "$batch_directory/shards" -name point_calibration.csv -print0 2>/dev/null)
    local now elapsed running=0 finished failed percent rate eta filled empty bar
    now=$(date +%s); elapsed=$((now-START_EPOCH))
    for file in "$batch_directory"/pids/*.pid; do
        [[ -e "$file" ]] && kill -0 "$(<"$file")" 2>/dev/null && running=$((running + 1))
    done
    finished=$(find "$batch_directory/status" -name '*.done' 2>/dev/null | wc -l)
    failed=$(find "$batch_directory/status" -name '*.failed' 2>/dev/null | wc -l)
    percent=$(awk -v n="$completed" -v d="$TOTAL_POINTS" 'BEGIN{printf "%.2f",100*n/d}')
    rate=$(awk -v n="$completed" -v s="$elapsed" 'BEGIN{if(s>0)printf "%.3f",n/s;else print "0.000"}')
    if (( completed > 0 )); then eta=$(awk -v n="$completed" -v d="$TOTAL_POINTS" -v s="$elapsed" 'BEGIN{printf "%.0f",(d-n)*s/n}'); else eta=0; fi
    filled=$(awk -v n="$completed" -v d="$TOTAL_POINTS" 'BEGIN{printf "%d",40*n/d}'); empty=$((40-filled))
    bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
    printf '[%s] %5s%% %d/%d points %.3f point/s elapsed %s ETA %s workers %d running, %d done, %d failed\n' \
        "$bar" "$percent" "$completed" "$TOTAL_POINTS" "$rate" "$(format_seconds "$elapsed")" "$(format_seconds "$eta")" "$running" "$finished" "$failed"
    (( completed >= TOTAL_POINTS )) && return 10
    (( running == 0 && finished + failed >= WORKERS )) && return 11
    return 0
}

action=${1:-}
case "$action" in
start)
    batch=${2:-}; workers=${3:-28}; toys=${4:-5000}; figures=${5:-both}
    case "$figures" in
        fig3a) total_grid_points=3721 ;;
        fig3b) total_grid_points=4514 ;;
        both) total_grid_points=8235 ;;
        *) echo "FIGURES must be fig3a, fig3b, or both" >&2; exit 2 ;;
    esac
    [[ "$batch" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] && (( workers <= total_grid_points )) || exit 2
    [[ "$toys" =~ ^[1-9][0-9]*$ ]] || exit 2
    [[ -x "$virtual_environment/bin/python" ]] || { echo "Missing environment: $virtual_environment" >&2; exit 2; }
    batch_directory="$output_root/$batch"; [[ ! -e "$batch_directory" ]] || { echo "Batch already exists: $batch_directory" >&2; exit 2; }
    mkdir -p "$batch_directory"/{logs,pids,status,shards}
    cat > "$batch_directory/manifest.env" <<EOF
BATCH=$batch
WORKERS=$workers
TOYS_PER_HYPOTHESIS=$toys
FIGURES=$figures
TOTAL_POINTS=$total_grid_points
START_EPOCH=$(date +%s)
GIT_COMMIT=$(git -C "$repository_root" rev-parse HEAD)
EOF
    for ((worker=0; worker<workers; worker++)); do
        start=$((worker*total_grid_points/workers)); stop=$(((worker+1)*total_grid_points/workers)); worker_name=$(printf 'worker_%02d' "$worker")
        nohup bash "$0" internal-worker "$batch" "$worker_name" "$start" "$stop" "$toys" "$figures" > "$batch_directory/logs/$worker_name.log" 2>&1 < /dev/null &
        echo $! > "$batch_directory/pids/$worker_name.pid"
    done
    echo "Started $workers workers for $total_grid_points points x $toys Toys per hypothesis."
    echo "Monitor: bash scripts/server/microboone_fig3b_reprofile_toy.sh watch $batch"
    ;;
internal-worker)
    batch=$2; worker_name=$3; start=$4; stop=$5; toys=$6; figures=$7; batch_directory="$output_root/$batch"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
    set +e
    "$virtual_environment/bin/python" -B "$repository_root/studies/microboone_fig3b_full_reprofile_toy/run.py" \
        --figures "$figures" --toys "$toys" --start-point "$start" --stop-point "$stop" --output-directory "$batch_directory/shards/$worker_name"
    code=$?; set -e
    if (( code == 0 )); then printf '0\n' > "$batch_directory/status/$worker_name.done"; else printf '%d\n' "$code" > "$batch_directory/status/$worker_name.failed"; fi
    exit "$code"
    ;;
status) batch_status "${2:?missing BATCH}" || code=$?; [[ ${code:-0} == 10 ]] && exit 0; exit "${code:-0}" ;;
watch)
    batch=${2:?missing BATCH}; interval=${3:-10}
    while true; do set +e; line=$(batch_status "$batch"); code=$?; set -e; printf '\r\033[K%s' "$line"; (( code == 10 )) && { printf '\nCompleted.\n'; break; }; (( code == 11 )) && { printf '\nStopped before completion.\n'; exit 1; }; sleep "$interval"; done
    ;;
stop)
    batch=${2:?missing BATCH}; batch_directory="$output_root/$batch"; [[ -d "$batch_directory/pids" ]] || exit 2
    for file in "$batch_directory"/pids/*.pid; do [[ -e "$file" ]] || continue; pid=$(<"$file"); command=$(ps -p "$pid" -o args= 2>/dev/null || true); [[ "$command" == *"internal-worker $batch"* ]] && kill "$pid"; done
    echo "Stop signal sent to $batch"
    ;;
*) usage; exit 2 ;;
esac
