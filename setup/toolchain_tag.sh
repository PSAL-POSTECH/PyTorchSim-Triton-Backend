#!/usr/bin/env bash
# The toolchain image's tag: the triton tag it is FROM, the one step it runs, and what that step
# builds -- the tree of the third_party/TOGSim submodule, its own submodule pins included.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -z "$(git -C third_party/TOGSim rev-parse --show-prefix 2>/dev/null || echo none)" ] || {
  echo "toolchain_tag.sh: third_party/TOGSim is not a checked-out submodule; run git submodule update --init third_party/TOGSim" >&2
  exit 1
}
{ setup/triton_tag.sh
  grep -hvE '^\s*(#|$)' setup/togsim.sh
  git -C third_party/TOGSim rev-parse 'HEAD^{tree}'
} | sha256sum | cut -c1-12
