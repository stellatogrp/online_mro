#!/bin/sh
# Build libworstcase next to the source.  No -ffast-math, and -ffp-contract=off
# (FMA contraction changes float semantics and breaks bit-parity of the
# bisection sign tests with the numpy oracle).
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" = "Darwin" ]; then
    # Apple clang: -march=native unsupported on arm64; OpenMP optional-skip.
    cc -O3 -ffp-contract=off -fPIC -shared -o libworstcase.dylib worstcase.c
else
    gcc -O3 -ffp-contract=off -fPIC -shared -march=native -fopenmp \
        -o libworstcase.so worstcase.c -lm
fi
echo "built $(ls libworstcase.* 2>/dev/null)"
