#!/usr/bin/env bash
set -euo pipefail
unset TRAIN_LIMIT TEST_LIMIT TRAIN_EPOCHS CHECK_WEIGHTS DUMP_PARAMETERS
root="$(cd "$(dirname "$0")/.." && pwd)"
output="$root/benchmark/results-$(date +%Y%m%d-%H%M%S)"
mkdir "$output"
{
    date -Is
    uname -a
    gcc --version | head -1
    mpicc --showme:version
    lscpu
    free -h
    sha256sum "$root/data/train-images-idx3-ubyte" "$root/data/train-labels-idx1-ubyte" "$root/data/t10k-images-idx3-ubyte" "$root/data/t10k-labels-idx1-ubyte"
} > "$output/environment.txt"
make -C "$root/sequential"
make -C "$root/parallel"
for run in 1 2 3; do
    (cd "$root/sequential" && ./digits_nn) | tee "$output/sequential-$run.log"
    (cd "$root/parallel" && mpirun -np 3 ./fcn_mpi) | tee "$output/mpi-$run.log"
done
printf 'Results: %s\n' "$output"
