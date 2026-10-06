#!/usr/bin/env bash
# The triton image's tag: only what decides that build -- the base it is FROM (the environment
# image, chosen by the submodule's pins), the distribution, the LLVM and triton pins and paths,
# and setup/triton.sh -- so a change to anything else does not rebuild triton.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
{ cat setup/system.sh setup/triton.sh
  grep -E '^(LLVM|TRITON)_' setup/versions.env
  git -C third_party/vcix-accelerator rev-parse HEAD:setup
} | sha256sum | cut -c1-12
