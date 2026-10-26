#!/usr/bin/env bash
# The triton image's tag: a hash of what decides that build and nothing else.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Comments and blank lines do not decide a build.
code() { grep -hvE '^\s*(#|$)' "$@"; }
{ code setup/system.sh setup/triton.sh
  # The versions those two scripts install: the triton stage is FROM the system stage, so cmake, conan
  # and nanobind are as much this build as LLVM and Triton are.
  grep -E '^(LLVM|TRITON|HOST_CMAKE|CONAN|NANOBIND)_' setup/versions.env
  git -C third_party/Coprocessor rev-parse HEAD:setup
} | sha256sum | cut -c1-12
