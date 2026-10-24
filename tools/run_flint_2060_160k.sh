#!/usr/bin/env bash
# Opt-in Flint reference profile for an RTX 2060 SUPER (8 GiB).
# Uses upstream llama.cpp until Strata's native Flint driver is ready.
set -euo pipefail

if (( $# < 1 )); then
    printf 'usage: STRATA_FLINT_LLAMA_CLI=/path/to/llama-cli %s MODEL.gguf [llama-cli arguments...]\n' "$0" >&2
    exit 2
fi

flint_model=$1
shift
flint_cli_bin=${STRATA_FLINT_LLAMA_CLI:-llama-cli}

exec "$flint_cli_bin" \
    -m "$flint_model" \
    -ngl 49 -ncmoe 40 -t 8 \
    -b 2048 -ub 1024 \
    -c 160000 -ctk q8_0 -ctv q4_0 -fa on \
    "$@"
