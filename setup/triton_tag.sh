#!/usr/bin/env bash
# The triton image's tag: a hash of what decides that build and nothing else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Comments and blank lines do not decide a build.
code() { grep -hvE '^\s*(#|$)' "$@"; }
{ code setup/system.sh setup/triton.sh
  grep -E '^(LLVM|TRITON)_' setup/versions.env
  git -C third_party/vcix-accelerator rev-parse HEAD:setup
} | sha256sum | cut -c1-12
