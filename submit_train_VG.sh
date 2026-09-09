#!/bin/bash

set -euo pipefail

CLUSTER="${1:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT_ARGS=()

case "${CLUSTER}" in
    helios)
        PARTITION="plgrid-gpu-gh200"
        ;;
    athena)
        ACCOUNT_ARGS=(--account=plgtabcfs-gpu-a100)
        PARTITION="plgrid-gpu-a100"
        ;;
    *)
        echo "Usage: $0 {helios|athena}" >&2
        exit 2
        ;;
esac

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

exec sbatch \
    "${ACCOUNT_ARGS[@]}" \
    --partition="${PARTITION}" \
    --export="ALL,TRAIN_CLUSTER=${CLUSTER}" \
    "${SCRIPT_DIR}/train_VG.sbatch"
