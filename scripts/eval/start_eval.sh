#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate internutopia

# CONFIG=scripts/eval/configs/h1_internvla_n1_cfg.py
export CUDA_VISIBLE_DEVICES=0,1
CONFIG=scripts/eval/configs/h1_rdp_cfg.py

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Extract the prefix from the config filename
CONFIG_BASENAME=$(basename "$CONFIG" .py)
CONFIG_PREFIX=$(echo "$CONFIG_BASENAME" | sed 's/_cfg$//')

# Create the logs directory if it doesn't exist
mkdir -p logs

# Set the log file paths
SERVER_LOG="logs/${CONFIG_PREFIX}_server.log"
EVAL_LOG="logs/${CONFIG_PREFIX}_eval.log"

# Only kill server processes started with the SAME config, avoid killing others
SERVER_PATTERN="vlnverse/agent/utils/server.py --config $CONFIG"
processes=$(pgrep -f "$SERVER_PATTERN")
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 "$pid"
        echo "kill server (same config): $pid"
    done
fi
python vlnverse/agent/utils/server.py --config $CONFIG > "$SERVER_LOG" 2>&1 &


RETRY_LIMIT=9999
MONITOR_INTERVAL=60
DEADLOCK_THRESHOLD=$((6 * 60))

START_COMMAND="python -u scripts/eval/eval.py --config $CONFIG"
LOG_FILE="$EVAL_LOG"

pid=0

retry_count=0

log_and_exit() {
    reason="$1"
    code="${2:-1}"
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $reason"
        echo "retries: $retry_count/$RETRY_LIMIT"
        if [ -n "$pid" ] && [ "$pid" -ne 0 ]; then
            echo "last known child pid: $pid"
        fi
    } >> "$LOG_FILE" 2>&1
    exit $code
}

start_process() {
    echo "Starting process..."
    
    # Kill and restart server.py
    SERVER_PATTERN="vlnverse/agent/utils/server.py --config $CONFIG"
    processes=$(pgrep -f "$SERVER_PATTERN")
    if [ -n "$processes" ]; then
        for pid_to_kill in $processes; do
            kill -9 "$pid_to_kill" 2>/dev/null
            echo "Killed server process: $pid_to_kill"
        done
    fi
    wait 2>/dev/null  # Wait for background jobs to finish and suppress messages
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting server.py..." >> "$SERVER_LOG"
    python vlnverse/agent/utils/server.py --config $CONFIG >> "$SERVER_LOG" 2>&1 &
    echo "Restarted server.py"
    
    # Start the main command
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting eval process (retry: $retry_count)..." >> "$LOG_FILE"
    $START_COMMAND >> "$LOG_FILE" 2>&1 &
    pid=$!
}


check_process() {
    if ! kill -0 $pid > /dev/null 2>&1; then
        echo "Process $pid has exited."
        return 1
    fi
    return 0
}


check_log_update() {
    if [ ! -e "$LOG_FILE" ]; then
        return 1
    fi
    last_update=$(stat -c %Y "$LOG_FILE")
    current_time=$(date +%s)

    delta=$(( current_time - last_update ))

    if [ $delta -ge $DEADLOCK_THRESHOLD ]; then
        echo "Log file has not been updated for $((DEADLOCK_THRESHOLD / 60)) minutes."
        return 1
    fi

    return 0
}

start_process

while true; do
    sleep $MONITOR_INTERVAL
    echo "start healthcheck"

    if ! check_process; then
        if [ $retry_count -lt $RETRY_LIMIT ]; then
            echo "Retrying... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
            retry_count=$((retry_count + 1))
            start_process
        else
            echo "Exceeded maximum retry attempts. Exiting."
            log_and_exit "Exceeded maximum retry attempts after process exit"
        fi
    else
        if ! check_log_update; then
            if [ $retry_count -lt $RETRY_LIMIT ]; then
                echo "Restarting process due to log file not updating... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
                retry_count=$((retry_count + 1))
                kill -9 $pid
                start_process
            else
                echo "Exceeded maximum retry attempts. Exiting."
                log_and_exit "Exceeded maximum retry attempts due to log not updating"
            fi
        fi
    fi
done
