#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$PROJECT_ROOT/logs/tensorboard_logs"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: TensorBoard log directory not found:"
    echo "$BASE_DIR"
    exit 1
fi

LATEST_EVENT=$(find "$BASE_DIR" -type f -name "events.out.tfevents.*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_EVENT" ]; then
    echo "Error: No TensorBoard event file found in:"
    echo "$BASE_DIR"
    exit 1
fi

RUN_DIR="$(dirname "$LATEST_EVENT")"

echo "Detected current active run:"
echo "$RUN_DIR"
echo

tensorboard --logdir="$RUN_DIR"
