#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 1 )); then
    echo "Usage: bash $0 INPUT_JSON_NAME [additional inference flags...]" >&2
    exit 1
fi
source "$SCRIPT_DIR/scripts/env.sh" || {
    echo "Create scripts/env.sh using scripts/env.sh.example before launching." >&2
    exit 1
}
cd -- "$SCRIPT_DIR"

INPUT_NAME="$1"
shift
source "$SCRIPT_DIR/run_preprocessing.sh"
configure_preprocessed_input || exit 1

exec "${XFOLD_INFERENCE_PYTHON:-python}" "$SCRIPT_DIR/run_alphafold.py" \
    --run_data_pipeline=False \
    "${INPUT_ARGS[@]}" \
    --model_dir="$MODEL_DIR" \
    --output_dir="$OUTPUT_DIR" \
    "$@"
