#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -shared -fPIC -o native_sc.so native_sc.c -lm
