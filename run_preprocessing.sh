#!/usr/bin/env bash

configure_preprocessed_input() {
    local root_dir output_root invocation_dir
    root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    if [[ -z "${XFOLD_PREPROCESSED_DIR:-}" ]]; then
        if [[ -z "${AF3_PREPROCESS_PYTHON:-}" || -z "${AF3_DB_DIR:-}" ]]; then
            echo "Automatic preprocessing requires AF3_PREPROCESS_PYTHON (AF3 3.0.4) and AF3_DB_DIR in scripts/env.sh." >&2
            return 1
        fi
        mkdir -p -- "$OUTPUT_DIR" || return 1
        output_root="$(cd -- "$OUTPUT_DIR" && pwd)" || return 1
        invocation_dir="$(mktemp -d "$output_root/preprocessing.XXXXXX")" || return 1
        XFOLD_PREPROCESSED_DIR="$invocation_dir/jobs"
        echo "Running accelerated preprocessing once before inference: $XFOLD_PREPROCESSED_DIR"
        if ! bash "$root_dir/run_preprocessing.sh" \
            "${JSON_PATH}/${INPUT_NAME}" "$XFOLD_PREPROCESSED_DIR"; then
            echo "Preprocessing failed; inference was not started. Output: $invocation_dir" >&2
            return 1
        fi
    fi
    if [[ ! -f "$XFOLD_PREPROCESSED_DIR/preprocessing_manifest.json" ]]; then
        echo "Missing completed preprocessing manifest in $XFOLD_PREPROCESSED_DIR; inference was not started." >&2
        return 1
    fi
    XFOLD_PREPROCESSED_DIR="$(cd -- "$XFOLD_PREPROCESSED_DIR" && pwd)" || return 1
    export XFOLD_PREPROCESSED_DIR
    INPUT_ARGS=(--preprocessed_dir="$XFOLD_PREPROCESSED_DIR")
}

run_preprocessing_main() {
    local root_dir input_json output_dir
    local -a options=()
    root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    if (( $# < 1 )); then
        echo "Usage: bash $0 INPUT_JSON [OUTPUT_DIR] [additional preprocessing flags...]" >&2
        return 1
    fi
    : "${AF3_PREPROCESS_PYTHON:?Set to the Python executable in a separate AF3 3.0.4 environment}"
    : "${AF3_DB_DIR:?Set to the directory containing the seven FASTA databases and templates}"
    input_json="$1"
    output_dir="${2:-$root_dir/preprocessed}"
    shift
    if (( $# > 0 )); then shift; fi

    [[ -z "${AF3_SEARCH_CPU_BUDGET:-}" ]] || options+=(--cpu_budget "$AF3_SEARCH_CPU_BUDGET")
    [[ -z "${AF3_JACKHMMER_N_CPU:-}" ]] || options+=(--jackhmmer_n_cpu "$AF3_JACKHMMER_N_CPU")
    [[ -z "${AF3_NHMMER_N_CPU:-}" ]] || options+=(--nhmmer_n_cpu "$AF3_NHMMER_N_CPU")
    [[ -z "${AF3_JACKHMMER_MAX_PARALLEL_SHARDS:-}" ]] || options+=(--jackhmmer_max_parallel_shards "$AF3_JACKHMMER_MAX_PARALLEL_SHARDS")
    [[ -z "${AF3_NHMMER_MAX_PARALLEL_SHARDS:-}" ]] || options+=(--nhmmer_max_parallel_shards "$AF3_NHMMER_MAX_PARALLEL_SHARDS")

    exec "$AF3_PREPROCESS_PYTHON" -u "$root_dir/run_preprocessing.py" \
        --json_path="$input_json" \
        --db_dir="$AF3_DB_DIR" \
        --shard_dir="${AF3_SHARD_DIR:-$AF3_DB_DIR/sharded}" \
        --num_shards="${AF3_DB_SHARDS:-2}" \
        --output_dir="$output_dir" \
        "${options[@]}" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    run_preprocessing_main "$@"
fi
