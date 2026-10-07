#!/usr/bin/env bash
# The toolchain image's tag: a hash of what decides that build, the triton tag included.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Comments and blank lines do not decide a build.
code() { grep -hvE '^\s*(#|$)' "$@"; }
{ code setup/versions.env setup/system.sh setup/toolchain.sh setup/Dockerfile
  git rev-parse HEAD:third_party/vcix-accelerator
  setup/triton_tag.sh
} | sha256sum | cut -c1-12
