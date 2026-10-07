#!/usr/bin/env bash
# The toolchain image's tag: the triton tag it is FROM, the one step it runs, and that step's pin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
{ setup/triton_tag.sh
  grep -hvE '^\s*(#|$)' setup/togsim.sh
  grep -E '^PYTORCHSIM_' setup/versions.env
} | sha256sum | cut -c1-12
