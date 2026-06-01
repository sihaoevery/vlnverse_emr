#!/bin/bash
# Chunked eval launcher — graceful mitigation for an Isaac/PhysX scene-collision leak
# (env.reset does not free the scene's static colliders; see project memory). Host RSS
# climbs every episode until a physics step stalls (OOM on low-RAM boxes, CPU-spin on
# high-RAM), so we run the eval in fresh processes and restart once RSS hits a budget —
# process death is what reclaims the native leak; resume-via-lmdb makes it seamless. A
# per-chunk inactivity watchdog catches a mid-episode stall. Stops cleanly when the eval
# reports "No more episodes". Does NOT touch start_eval_one_gpu.sh.
#
# Usage: scripts/eval/start_eval_chunked.sh --config <cfg.py>
#   Tunables (env): VLN_RSS_BUDGET_MIB (restart threshold, MiB),
#                   VLN_INACTIVITY_TIMEOUT (mid-episode-stall kill, s),
#                   VLN_CHUNK_HARDCAP (per-chunk wall-clock backstop, s).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIG=scripts/eval/configs/h1_cma_clip_cfg_vlnverse_coarse.py

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

CONFIG_BASENAME=$(basename "$CONFIG" .py)
CONFIG_PREFIX=$(echo "$CONFIG_BASENAME" | sed 's/_cfg$//')
mkdir -p logs
SERVER_LOG="logs/${CONFIG_PREFIX}_server.log"
EVAL_LOG="logs/${CONFIG_PREFIX}_eval.log"

# Restart trigger: exit a chunk once host RSS exceeds this budget, before the accumulated
# scene collision stalls a physics step. The 24 GB default is tuned for our setup (128 GB
# RAM + RTX 4090, where the stall appears around ~28 GB RSS, and a single big-scene episode
# can jump RSS ~5 GB mid-step) and gave ~1.5x throughput vs a conservative budget. TUNE TO
# YOUR OWN RAM: lower it for more safety margin, raise it for fewer (riskier) Isaac reboots.
# Auto-capped to fit free RAM on low-memory boxes.
if [ -z "${VLN_RSS_BUDGET_MIB:-}" ]; then
    free_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    VLN_RSS_BUDGET_MIB=24000
    cap=$(( free_mib - 6000 ))
    [ "$VLN_RSS_BUDGET_MIB" -gt "$cap" ] && VLN_RSS_BUDGET_MIB=$cap
    [ "$VLN_RSS_BUDGET_MIB" -lt 14000 ] && VLN_RSS_BUDGET_MIB=14000
fi
export VLN_RSS_BUDGET_MIB
echo "RSS budget: ${VLN_RSS_BUDGET_MIB} MiB/run (restart before physics-step stall)"

MAX_CHUNKS=200
chunk=0
while [ $chunk -lt $MAX_CHUNKS ]; do
    chunk=$((chunk + 1))
    echo "[$(date '+%F %T')] === chunk $chunk, config=$CONFIG ==="

    pkill -9 -f "server.py --config $CONFIG" 2>/dev/null
    sleep 8  # let the prior chunk's GPU/CUDA context fully release before re-init
    echo "[$(date '+%F %T')] start server (chunk $chunk)" >> "$SERVER_LOG"
    python vlnverse/agent/utils/server.py --config "$CONFIG" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    sleep 3

    CHUNK_OUT="${EVAL_LOG}.chunk"
    echo "[$(date '+%F %T')] start eval (chunk $chunk)" >> "$EVAL_LOG"
    # Run eval in the background and watch it with an INACTIVITY watchdog: a mid-episode
    # PhysX stall makes stdout go silent (startup/episodes stream continuously), so if
    # CHUNK_OUT's mtime is stale for INACTIVITY_TIMEOUT we kill+resume in minutes. The
    # wall-clock hardcap below is only a paranoia backstop for some unforeseen runaway —
    # healthy chunks are minutes (p99 ~40 min), so 6 h must never false-kill a real one.
    python -u scripts/eval/eval.py --config "$CONFIG" > "$CHUNK_OUT" 2>&1 &
    EPID=$!
    chunk_start=$(date +%s); killed_hung=0
    while kill -0 "$EPID" 2>/dev/null; do
        sleep 20
        now=$(date +%s)
        mt=$(stat -c %Y "$CHUNK_OUT" 2>/dev/null || echo "$now")
        if [ $(( now - mt )) -ge "${VLN_INACTIVITY_TIMEOUT:-420}" ]; then
            echo "[$(date '+%F %T')] chunk $chunk HUNG (stdout silent $(( now - mt ))s) -> kill+resume" | tee -a "$EVAL_LOG"
            killed_hung=1; break
        fi
        if [ $(( now - chunk_start )) -ge "${VLN_CHUNK_HARDCAP:-21600}" ]; then
            echo "[$(date '+%F %T')] chunk $chunk hit hard cap -> kill+resume" | tee -a "$EVAL_LOG"
            killed_hung=1; break
        fi
    done
    if [ "$killed_hung" -eq 1 ]; then
        kill -9 "$EPID" 2>/dev/null
        pkill -9 -f "eval.py --config $CONFIG" 2>/dev/null
        rc=137
    else
        wait "$EPID"; rc=$?
    fi
    cat "$CHUNK_OUT" >> "$EVAL_LOG"

    kill -9 $SERVER_PID 2>/dev/null
    pkill -9 -f "server.py --config $CONFIG" 2>/dev/null

    if grep -q "No more episodes to evaluate" "$CHUNK_OUT"; then
        echo "[$(date '+%F %T')] all episodes done (chunk $chunk). Stopping."
        rm -f "$CHUNK_OUT"
        exit 0
    fi
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] chunk $chunk eval exited rc=$rc (crash before limit); relaunching to resume."
    fi
done

echo "[$(date '+%F %T')] Hit MAX_CHUNKS=$MAX_CHUNKS without completion; stopping."
exit 1
