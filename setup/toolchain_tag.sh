#!/usr/bin/env bash
# The toolchain image's tag: the triton tag it is FROM, the one step it runs, and what that step
# builds -- the TOGSim tree of the third_party/PyTorchSim submodule, its own submodule pins included.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
{ setup/triton_tag.sh
  grep -hvE '^\s*(#|$)' setup/togsim.sh
  git -C third_party/PyTorchSim rev-parse HEAD:TOGSim
} | sha256sum | cut -c1-12
