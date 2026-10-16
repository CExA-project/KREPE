#!/bin/bash

set -o pipefail
shopt -s nullglob

if [ $# -ne 6 ]; then
  echo "The script expects 6 arguments"
  exit 1
fi

LIBKREPE_PATH=$1
EXECUTABLE=$2
REPLAY_EXECUTABLE=$3
COMPARE_OUTPUT=0
if [ "${4:-}" = "TRUE" ]; then
  COMPARE_OUTPUT=1
fi
EXTRA_KOKKOS_TOOLS_ARGS=$5
REPLAYER_ARGS=$6

OUTPUT_DIR=$(mktemp -d)
REFERENCE_OUTPUT="$OUTPUT_DIR/reference.txt"
REPLAY_OUTPUT="$OUTPUT_DIR/replay.txt"

clean_exit() {
  rm -rf "$OUTPUT_DIR"
  exit $1
}

# TODO: Make output comparison less dependent on Kokkos/runtime log formatting.
# Prefer comparing only stable test result lines instead of filtering known log noise.
clean_output() {
  grep -vF -- "[krepe-capture]" "$1" \
    | grep -vF -- "KokkosP:" \
    | grep -vF -- "-----------------------------------------------------------"
}

rm -f -- *.h5

KOKKOS_TOOLS_LIBS="$LIBKREPE_PATH" KOKKOS_TOOLS_ARGS="--krepe-dump-kernel-label=test_kernel $EXTRA_KOKKOS_TOOLS_ARGS" $EXECUTABLE | tee "$REFERENCE_OUTPUT"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  clean_exit $EXIT_CODE
fi

DUMP_FILES=(krepe_*.h5)
if (( ${#DUMP_FILES[@]} != 1 )); then
  echo "Expected exactly one HDF5 dump, found ${#DUMP_FILES[@]}"
  clean_exit 1
fi

$REPLAY_EXECUTABLE --kernel-replayer-dump="${DUMP_FILES[0]}" $REPLAYER_ARGS | tee "$REPLAY_OUTPUT"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  clean_exit $EXIT_CODE
fi

if (( COMPARE_OUTPUT == 0 )); then
  clean_exit 0
fi

CLEAN_REFERENCE_OUTPUT="$OUTPUT_DIR/clean_reference.txt"
CLEAN_REPLAY_OUTPUT="$OUTPUT_DIR/clean_replay.txt"
clean_output "$REFERENCE_OUTPUT" > "$CLEAN_REFERENCE_OUTPUT"
clean_output "$REPLAY_OUTPUT" > "$CLEAN_REPLAY_OUTPUT"
if ! diff "$CLEAN_REFERENCE_OUTPUT" "$CLEAN_REPLAY_OUTPUT"; then
  clean_exit 1
fi
